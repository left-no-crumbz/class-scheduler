import flet as ft

import main_rewrite as mr

# Lists to store created instructor and subject objects
instructor_obj = []
subject_obj = []


def main(page: ft.Page):
    # Input fields for instructors and subjects
    instructor_name = ft.TextField(label="Instructor Name")
    subject_name = ft.TextField(label="Subject Name")
    instructor_checkboxes_view = ft.ListView()
    instructor_checkboxes = []

    subject_type_dropdown = ft.Dropdown(
        options=[
            ft.dropdown.Option("SPECIALIZED_LECTURE"),
            ft.dropdown.Option("SPECIALIZED_LAB"),
            ft.dropdown.Option("GENERAL_EDUCATION"),
        ],
        label="Subject Type",
    )

    instructor_list_view = ft.ListView()
    subject_list_view = ft.ListView()

    def create_subject_view():
        page.update()

        # instructor_obj should not be empty if an instructor is added
        instructor_checkboxes_view.controls.clear()
        instructor_checkboxes.clear()

        for instructor in instructor_obj:
            # instructor_checkboxes_view.controls.append(
            #     ft.Checkbox(label=instructor.get_name())
            # )
            instructor_checkboxes.append(ft.Checkbox(label=instructor.get_name()))

        instructor_checkboxes_view.controls.extend(instructor_checkboxes)

        page.update()

        return instructor_checkboxes_view

    # Function to add an instructor
    def add_instructor(e):
        if not instructor_name.value:
            page.snack_bar = ft.SnackBar(
                content=ft.Text("Instructor name cannot be blank!")
            )
            page.snack_bar.open = True
            page.update()
        elif instructor_name.value:
            instructor = mr.create_intructor(instructor_name.value)
            instructor_obj.append(instructor)

            # Update the instructor list view
            instructor_list_view.controls.append(
                ft.Card(ft.ListTile(title=ft.Text(instructor_name.value)))
            )

            # Update the checkboxes after adding a new instructor
            create_subject_view()

            instructor_name.value = ""
            page.update()

    # Function to add a subject
    def add_subject(e):
        if not subject_name.value:
            page.snack_bar = ft.SnackBar(
                content=ft.Text("Subject name cannot be blank!")
            )
            page.snack_bar.open = True
            page.update()
            return

        if not subject_type_dropdown.value:
            page.snack_bar = ft.SnackBar(
                content=ft.Text("Please select a subject type!")
            )
            page.snack_bar.open = True
            page.update()
            return

        # Gather selected instructors
        # FIXME:
        selected_instructors = [
            instructor_obj[i]
            for i, checkbox in enumerate(instructor_checkboxes)
            if checkbox.value
        ]

        for i, checkbox in enumerate(instructor_checkboxes):
            print(checkbox.value)

        print(f"Selected Instructors: {selected_instructors}")
        print(f"Instructor Checkboxes: {instructor_checkboxes}")
        print(f"Instructor Checkboxes: {instructor_checkboxes}")

        if not selected_instructors:
            page.snack_bar = ft.SnackBar(
                content=ft.Text("Please select at least one instructor!")
            )
            page.snack_bar.open = True
            page.update()
            return

        subject_type = getattr(mr.SubjectType, subject_type_dropdown.value)
        subject = mr.create_subject(
            subject_name.value, selected_instructors, subject_type
        )
        print(subject.get_subject_name())
        print(subject.get_subject_instructors())
        print(subject._subject_type)
        subject_obj.append(subject)

        # Add the created subject to the subject list view
        subject_list_view.controls.append(
            ft.Card(ft.ListTile(title=ft.Text(subject_name.value)))
        )

        # Reset input fields
        subject_name.value = ""
        subject_type_dropdown.value = None
        for checkbox in instructor_checkboxes:
            checkbox.value = False
        page.update()

    # Views for creating subjects and instructors
    create_subjects_view = ft.View(
        "/create_subjects",
        controls=[
            ft.SafeArea(
                ft.ListView(
                    controls=[
                        ft.Text(
                            "SchedTool",
                            style=ft.TextStyle(size=24, weight=ft.FontWeight.BOLD),
                        ),
                        ft.Text(
                            "Step 2 of 5",
                            style=ft.TextStyle(color=ft.colors.GREY),
                        ),
                        ft.Container(height=16),
                        ft.Text(
                            "Create Subjects",
                            style=ft.TextStyle(size=24, weight=ft.FontWeight.BOLD),
                        ),
                        ft.Container(height=8),
                        subject_name,
                        ft.Container(height=8),
                        subject_type_dropdown,
                        ft.Container(height=8),
                        ft.Text("Select Instructors:"),
                        create_subject_view(),  # This will dynamically generate the checkboxes
                        ft.Container(height=8),
                        ft.ElevatedButton(text="Add Subject", on_click=add_subject),
                        ft.Container(height=8),
                        ft.Text("Added Subjects:"),
                        subject_list_view,
                    ]
                ),
                minimum_padding=32,
            )
        ],
    )

    # Navigation to create subjects page
    def next_page(e):
        page.views.append(create_subjects_view)
        page.go("/create_subjects")

    # Main page to add instructors
    page.add(
        ft.SafeArea(
            ft.ListView(
                controls=[
                    ft.Text(
                        "SchedTool",
                        style=ft.TextStyle(size=24, weight=ft.FontWeight.BOLD),
                    ),
                    ft.Text(
                        "Step 1 of 5",
                        style=ft.TextStyle(color=ft.colors.GREY),
                    ),
                    ft.Container(height=16),
                    ft.Text(
                        "Create Instructors",
                        style=ft.TextStyle(weight=ft.FontWeight.BOLD),
                    ),
                    ft.Container(height=8),
                    instructor_name,
                    ft.Container(height=8),
                    ft.ElevatedButton(text="Add Instructor", on_click=add_instructor),
                    ft.Container(height=8),
                    ft.Text("Added Instructors:"),
                    ft.Container(height=8),
                    instructor_list_view,
                    ft.Container(height=8),
                    ft.ElevatedButton(text="Next", on_click=next_page),
                ]
            ),
            minimum_padding=32,
        )
    )


ft.app(main)
