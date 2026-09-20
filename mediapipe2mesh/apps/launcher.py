"""Python launcher with independent preview and multithread configurations."""
import argparse
from pathlib import Path

from mediapipe2mesh.config import CONFIG_ROOT, load_config


MODE_CONFIGS = {'interaction': 'interaction.json', 'multicore': 'multicore.json',
                'viewer': 'viewer.json'}


def mode_config(mode, override=None):
    return load_config(MODE_CONFIGS[mode], override or None)


def validate_mode(mode, config):
    from mediapipe2mesh.apps.viewer_app import validate_config as validate_viewer
    from mediapipe2mesh.apps.interaction_app import validate_config as validate_interaction
    from mediapipe2mesh.tracking.filters import OneEuroFilter
    validate_viewer(config)
    if mode != 'viewer':
        validate_interaction(config)
    workers = config.mano.jacobian_workers
    if isinstance(workers, bool) or not isinstance(workers, int) or not 0 <= workers <= 45:
        raise ValueError('mano.jacobian_workers must be an integer in [0,45]')
    if config.mano.jacobian_backend not in ('thread', 'process'):
        raise ValueError('mano.jacobian_backend must be thread or process')
    OneEuroFilter(**config.tracking.position_filter.as_dict())


def run_mode(mode, override=None):
    config = mode_config(mode, override)
    validate_mode(mode, config)
    if mode == 'viewer':
        from mediapipe2mesh.apps.viewer_app import run_viewer
        run_viewer(config)
    else:
        from mediapipe2mesh.apps.interaction_app import run_interaction
        run_interaction(config)


def show_launcher():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    root = tk.Tk()
    root.title('MediaPipe → MANO')
    root.geometry('420x210')
    overrides = {'interaction': '', 'multicore': ''}
    frame = ttk.Frame(root, padding=20)
    frame.pack(fill='both', expand=True)
    ttk.Label(frame, text='选择运行模式').pack(anchor='w', pady=(0, 12))

    def start(mode):
        try:
            config = mode_config(mode, overrides[mode])
            validate_mode(mode, config)
            root.withdraw()
            root.update_idletasks()
            run_mode(mode, overrides[mode])
        except Exception as error:
            messagebox.showerror('运行失败', str(error), parent=root)
        finally:
            root.deiconify()

    def choose_config():
        dialog = tk.Toplevel(root)
        dialog.title('选择配置')
        dialog.transient(root)
        panel = ttk.Frame(dialog, padding=16)
        panel.pack(fill='both', expand=True)
        entries = {}
        for row, (mode, title) in enumerate((('interaction', '手部交互预览'),
                                              ('multicore', '多线程调用测试'))):
            entries[mode] = tk.StringVar(value=overrides[mode])
            ttk.Label(panel, text=title).grid(row=row * 2, column=0, padx=5, pady=8)
            ttk.Entry(panel, textvariable=entries[mode], width=58).grid(row=row * 2, column=1)

            def browse(selected=mode):
                file = filedialog.askopenfilename(parent=dialog, title='选择配置',
                    initialdir=str(CONFIG_ROOT), filetypes=[('JSON', '*.json')])
                if file:
                    entries[selected].set(file)

            ttk.Button(panel, text='浏览', command=browse).grid(row=row * 2, column=2, padx=5)
            ttk.Label(panel, text='留空使用 ' + str(CONFIG_ROOT / MODE_CONFIGS[mode])).grid(
                row=row * 2 + 1, column=1, sticky='w')

        def save():
            try:
                selected = {mode: value.get().strip() for mode, value in entries.items()}
                for mode, file in selected.items():
                    validate_mode(mode, mode_config(mode, file))
                overrides.update(selected)
                dialog.destroy()
            except Exception as error:
                messagebox.showerror('配置错误', str(error), parent=dialog)

        buttons = ttk.Frame(panel)
        buttons.grid(row=4, column=0, columnspan=3, pady=(16, 0))
        ttk.Button(buttons, text='确定', command=save).pack(side='left', padx=5)
        ttk.Button(buttons, text='取消', command=dialog.destroy).pack(side='left', padx=5)
        dialog.grab_set()

    ttk.Button(frame, text='手部交互预览', command=lambda: start('interaction')).pack(fill='x', pady=4)
    ttk.Button(frame, text='多线程调用测试', command=lambda: start('multicore')).pack(fill='x', pady=4)
    ttk.Button(frame, text='选择配置', command=choose_config).pack(fill='x', pady=4)
    root.mainloop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config_path', nargs='?', help='JSON override for the selected mode')
    parser.add_argument('--mode', choices=tuple(MODE_CONFIGS))
    parser.add_argument('--config', help='JSON override for the selected mode')
    args = parser.parse_args(argv)
    if args.config and args.config_path:
        parser.error('Use either the positional config or --config')
    override = args.config or args.config_path
    if args.mode or override:
        run_mode(args.mode or 'interaction', str(Path(override).resolve()) if override else None)
    else:
        show_launcher()


if __name__ == '__main__':
    main()
