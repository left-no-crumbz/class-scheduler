import flet as ft


def main(page: ft.Page):
    def add_step(e):
        pass

    def subtract_step(e):
        pass

    page.padding = ft.padding.all(32)
    page.add(
        ft.SafeArea(
            ft.Column(
                controls=[
                    ft.Text(
                        "Schedulify",
                        style=ft.TextStyle(size=24, weight=ft.FontWeight.BOLD),
                    ),
                    ft.Text(
                        "Step 1 of 5", style=ft.TextStyle(color=ft.colors.GREY, size=14)
                    ),  # refactor to be stateful
                    ft.Container(height=16),
                    ft.Text("Create Instructors"),
                    ft.Container(height=8),
                    ft.TextField(label="Instructor Name"),
                    ft.Container(height=8),
                    ft.ElevatedButton(
                        text="Add Instructor", on_click=add_step
                    ),  # change this later
                ]
            )
        )
    )


ft.app(main)
