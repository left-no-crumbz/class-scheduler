import flet as ft

import main_rewrite as mr

# Lists to store created instructor and subject objects
instructor_obj = []
subject_obj = []
room_obj = []


def main(page: ft.Page):
    # Input fields for instructors and subjects
    instructor_name_field = ft.TextField(label="Instructor Name")
    subject_name_field = ft.TextField(label="Subject Name")
    room_number_field = ft.TextField(label="Room Number")

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

    room_type_dropdown = ft.Dropdown(
        options=[
            ft.dropdown.Option("LAB"),
            ft.dropdown.Option("LECTURE"),
        ],
        label="Room Type",
    )

    instructor_list_view = ft.ListView()
    subject_list_view = ft.ListView()
    room_list_view = ft.ListView()

    # ######### NAVIGATION ############
    # Navigation to create subjects page
    def go_to_create_subjects(e):
        page.views.append(create_subjects_view)
        page.go("/create_subjects")

    def go_to_create_rooms(e):
        page.views.append(create_room_view)
        page.go("/create_room")

    def go_back_to_instructors(e):
        page.views.pop()
        page.update()

    # ######### NAVIGATION ############

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
        if not instructor_name_field.value:
            page.snack_bar = ft.SnackBar(
                content=ft.Text("Instructor name cannot be blank!")
            )
            page.snack_bar.open = True
            page.update()
        elif instructor_name_field.value:
            instructor = mr.create_intructor(instructor_name_field.value)
            instructor_obj.append(instructor)

            # Update the instructor list view
            instructor_list_view.controls.append(
                ft.Card(ft.ListTile(title=ft.Text(instructor_name_field.value)))
            )

            # Update the checkboxes after adding a new instructor
            create_subject_view()

            instructor_name_field.value = ""
            page.update()

    # Function to add a room
    def add_room(e):
        if not room_number_field.value:
            page.snack_bar = ft.SnackBar(
                content=ft.Text("Room number cannot be blank!")
            )
            page.snack_bar.open = True
            page.update()
        if not room_type_dropdown.value:
            page.snack_bar = ft.SnackBar(content=ft.Text("Please select a room type!"))
            page.snack_bar.open = True
            page.update()
            return

        if room_number_field.value:
            room_type = getattr(mr.RoomType, room_type_dropdown.value)

            room = mr.create_room(room_number_field.value, room_type)

            room_obj.append(room)

            # Update the instructor list view
            room_list_view.controls.append(
                ft.Card(
                    ft.ListTile(
                        title=ft.Text(
                            f"{room.get_room()} - {room.get_room_type().name}"
                        )
                    )
                )
            )

            room_number_field.value = ""
            room_type_dropdown.value = None
            page.update()

    # Function to add a subject
    def add_subject(e):
        if not subject_name_field.value:
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
        selected_instructors = [
            instructor_obj[i]
            for i, checkbox in enumerate(instructor_checkboxes)
            if checkbox.value
        ]

        for i, checkbox in enumerate(instructor_checkboxes):
            print(checkbox.value)

        if not selected_instructors:
            page.snack_bar = ft.SnackBar(
                content=ft.Text("Please select at least one instructor!")
            )
            page.snack_bar.open = True
            page.update()
            return

        subject_type = getattr(mr.SubjectType, subject_type_dropdown.value)
        subject = mr.create_subject(
            subject_name_field.value, selected_instructors, subject_type
        )
        print(subject.get_subject_name())
        print(subject.get_subject_instructors())
        print(subject._subject_type)
        subject_obj.append(subject)

        # Add the created subject to the subject list view
        subject_list_view.controls.append(
            ft.Card(
                ft.ListTile(
                    title=ft.Text(
                        f"{subject_name_field.value} - {[instructor.get_name() for instructor in subject.get_subject_instructors()]} - {subject._subject_type.name}"
                    )
                )
            )
        )

        # Reset input fields
        subject_name_field.value = ""
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
                            style=ft.TextStyle(weight=ft.FontWeight.BOLD),
                        ),
                        ft.Container(height=8),
                        subject_name_field,
                        ft.Container(height=8),
                        subject_type_dropdown,
                        ft.Container(height=8),
                        ft.Text("Select Instructors:"),
                        create_subject_view(),  # This will dynamically generate the checkboxes
                        ft.Container(height=16),
                        ft.Row(
                            controls=[
                                ft.ElevatedButton(
                                    text="Add Subject",
                                    on_click=add_subject,
                                    width=150,
                                    height=35,
                                ),
                            ]
                        ),
                        ft.Container(height=16),
                        ft.Text("Added Subjects:"),
                        ft.Container(height=8),
                        subject_list_view,
                        ft.Container(height=16),
                        ft.Row(
                            controls=[
                                ft.Container(width=16),
                                ft.ElevatedButton(
                                    text="Previous",
                                    on_click=go_back_to_instructors,
                                    width=120,
                                    height=35,
                                ),
                                ft.Container(expand=True, expand_loose=True),
                                ft.ElevatedButton(
                                    text="Next",
                                    on_click=go_to_create_rooms,
                                    width=80,
                                    height=35,
                                ),
                                ft.Container(width=16),
                            ]
                        ),
                    ]
                ),
                minimum_padding=32,
            )
        ],
    )

    create_subjects_view.scroll = ft.ScrollMode.ALWAYS

    # Views for creating subjects and instructors
    create_room_view = ft.View(
        "/create_room",
        controls=[
            ft.SafeArea(
                ft.ListView(
                    controls=[
                        ft.Text(
                            "SchedTool",
                            style=ft.TextStyle(size=24, weight=ft.FontWeight.BOLD),
                        ),
                        ft.Text(
                            "Step 3 of 5",
                            style=ft.TextStyle(color=ft.colors.GREY),
                        ),
                        ft.Container(height=16),
                        ft.Text(
                            "Create Room",
                            style=ft.TextStyle(weight=ft.FontWeight.BOLD),
                        ),
                        ft.Container(height=8),
                        room_number_field,
                        ft.Container(height=8),
                        room_type_dropdown,
                        ft.Container(height=8),
                        ft.Row(
                            controls=[
                                ft.ElevatedButton(
                                    text="Add Room",
                                    on_click=add_room,
                                    width=120,
                                    height=40,
                                ),
                            ]
                        ),
                        ft.Container(height=8),
                        ft.Text("Added Rooms:"),
                        room_list_view,
                        ft.Container(height=8),
                        ft.Row(
                            controls=[
                                ft.Container(width=16),
                                ft.ElevatedButton(
                                    text="Previous",
                                    on_click=go_back_to_instructors,
                                    width=120,
                                    height=35,
                                ),
                                ft.Container(expand=True, expand_loose=True),
                                ft.ElevatedButton(
                                    text="Next",
                                    on_click=go_to_create_rooms,
                                    width=80,
                                    height=35,
                                ),
                                ft.Container(width=16),
                            ]
                        ),
                    ]
                ),
                minimum_padding=32,
            )
        ],
    )

    create_room_view.scroll = ft.ScrollMode.ALWAYS

    page.scroll = ft.ScrollMode.ALWAYS

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
                    instructor_name_field,
                    ft.Container(height=16),
                    ft.Row(
                        controls=[
                            ft.ElevatedButton(
                                text="Add Instructor",
                                on_click=add_instructor,
                                width=150,
                                height=35,
                            ),
                        ]
                    ),
                    ft.Container(height=16),
                    ft.Text("Added Instructors:"),
                    ft.Container(height=8),
                    instructor_list_view,
                    ft.Container(height=8),
                    ft.Row(
                        controls=[
                            ft.ElevatedButton(
                                text="Next",
                                on_click=go_to_create_subjects,
                                width=80,
                                height=35,
                            ),
                        ]
                    ),
                ],
            ),
            minimum_padding=32,
        )
    )


ft.app(main)
