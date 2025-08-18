import asyncio

import click
from jims_core.app import JimsApp

from jims_tui.chat_app import ChatApp


def load_jims_app(app_name: str) -> JimsApp:
    app_split = app_name.split(":")

    if len(app_split) == 1:
        module_name = app_split[0]
        app_attr = "app"
    elif len(app_split) == 2:
        module_name, app_attr = app_split
    else:
        raise Exception(f"Expected APP in format 'module:app' got '{app_name}'")

    from importlib import import_module

    # sys.path.append(os.getcwd())

    app_mod = import_module(module_name)
    app = getattr(app_mod, app_attr)

    assert isinstance(app, JimsApp)

    return app


@click.command()
@click.option("--app", type=click.STRING, default="app")
def cli(app: str) -> None:
    jims_app = load_jims_app(app)

    async def run_chat_app():
        chat_app = await ChatApp.create(jims_app)
        await chat_app.run_async()

    asyncio.run(run_chat_app())


if __name__ == "__main__":
    cli(auto_envvar_prefix="JIMS_TUI")
