import os

import flet as ft

import main_rewrite as mr

# Lists to store created instructor and subject objects
instructor_obj = []
subject_obj = []
room_obj = []
dept_obj = []
block_obj = []


def main(page: ft.Page):
    # Input fields for instructors and subjects
    instructor_name_field = ft.TextField(label="Instructor Name")
    subject_name_field = ft.TextField(label="Subject Name")
    room_number_field = ft.TextField(label="Room Number")
    block_number_field = ft.TextField(label="Block Number")
    population_size_field = ft.TextField(
        label="Population Size", keyboard_type=ft.KeyboardType.NUMBER
    )
    subpopulation_size_field = ft.TextField(
        label="Subpopulation Size", keyboard_type=ft.KeyboardType.NUMBER
    )
    migration_interval_field = ft.TextField(
        label="Migration Interval", keyboard_type=ft.KeyboardType.NUMBER
    )
    generation_field = ft.TextField(
        label="Number of Generations", keyboard_type=ft.KeyboardType.NUMBER
    )

    migration_rate_text = ft.Text()
    elitism_rate_text = ft.Text()

    migration_rate_text.value = "Migration Rate: 0.1"
    elitism_rate_text.value = "Elitism Rate: 0.1"

    loading_indicator = ft.ProgressRing(visible=False)
    images_container = ft.Column()  # To show generated schedule images

    def slider_changed_migration(e):
        migration_rate_text.value = f"Migration Rate: {e.control.value}"
        page.update()

    def slider_changed_elitism(e):
        elitism_rate_text.value = f"Elitism Rate: {e.control.value}"
        page.update()

    migration_rate_slider = ft.Slider(
        min=0.1,
        max=1,
        divisions=100,
        label="{value}%",
        on_change=slider_changed_migration,
    )
    elitism_rate_slider = ft.Slider(
        min=0.1,
        max=1,
        divisions=100,
        label="{value}%",
        on_change=slider_changed_elitism,
    )

    instructor_checkboxes_view = ft.ListView()
    subject_checkboxes_view = ft.ListView()

    instructor_checkboxes = []
    subject_checkboxes = []

    subject_type_dropdown = ft.Dropdown(
        options=[
            ft.dropdown.Option("SPECIALIZED_LECTURE"),
            ft.dropdown.Option("SPECIALIZED_LAB"),
            ft.dropdown.Option("GENERAL_EDUCATION"),
        ],
        label="Select Subject Type",
    )

    room_type_dropdown = ft.Dropdown(
        options=[
            ft.dropdown.Option("LAB"),
            ft.dropdown.Option("LECTURE"),
        ],
        label="Select Room Type",
    )

    course_dropdown = ft.Dropdown(
        options=[
            ft.dropdown.Option("CS"),
            ft.dropdown.Option("WD"),
            ft.dropdown.Option("NA"),
            ft.dropdown.Option("EMC"),
            ft.dropdown.Option("CYB"),
        ],
        label="Select Course",
    )

    year_level_dropdown = ft.Dropdown(
        options=[
            ft.dropdown.Option("FIRST"),
            ft.dropdown.Option("SECOND"),
            ft.dropdown.Option("THIRD"),
            ft.dropdown.Option("FOURTH"),
        ],
        label="Select Year Level",
    )

    department_dropdown = ft.Dropdown(options=[], label="Select Department")

    instructor_list_view = ft.ListView()
    subject_list_view = ft.ListView()
    room_list_view = ft.ListView()
    dept_list_view = ft.ListView()
    block_list_view = ft.ListView()

    # ######### NAVIGATION ############
    # Navigation to create subjects page
    def go_to_create_subjects(e):
        page.views.append(create_subjects_view)
        page.go("/create_subjects")

    def go_to_create_rooms(e):
        page.views.append(create_room_view)
        page.go("/create_room")

    def go_to_create_departments(e):
        page.views.append(create_depts_view)
        page.go("/create_dept")

    def go_to_create_block(e):
        page.views.append(create_block_view)
        page.go("/create_block")

    def go_to_create_config(e):
        page.views.append(create_config_view)
        page.go("/create_config")

    def go_back_to_instructors(e):
        page.views.pop()
        page.update()

    # ######### NAVIGATION ############

    def check_images_available(block_count):
        """Check if the generated schedule images are available."""
        available_images = []
        for i in range(1, block_count + 1):
            image_path = f"./block{i}_schedule.png"
            if os.path.exists(image_path):
                available_images.append(image_path)
        return available_images

    def create_subject_view():
        page.update()

        # instructor_obj should not be empty if an instructor is added
        instructor_checkboxes_view.controls.clear()
        instructor_checkboxes.clear()

        for instructor in instructor_obj:
            instructor_checkboxes.append(ft.Checkbox(label=instructor.get_name()))

        instructor_checkboxes_view.controls.extend(instructor_checkboxes)

        page.update()

        return instructor_checkboxes_view

    def create_dept_view():
        page.update()

        # instructor_obj should not be empty if an instructor is added
        subject_checkboxes_view.controls.clear()
        subject_checkboxes.clear()

        for subject in subject_obj:
            subject_checkboxes.append(ft.Checkbox(label=subject.get_subject_name()))

        subject_checkboxes_view.controls.extend(subject_checkboxes)

        page.update()

        return subject_checkboxes_view

    # Function to add an instructor
    def add_instructor(e):
        def delete_instructor(instructor):
            # Find the index of the instructor in instructor_obj
            index = instructor_obj.index(instructor)

            if index != -1:
                # Remove the instructor from both instructor_obj and the visual list view
                del instructor_obj[index]
                del instructor_list_view.controls[index]
                print(instructor_obj)
                instructor_list_view.update()  # Update the list view to reflect changes
                create_subject_view()
                page.update()  # Update the page to reflect changes

        if not instructor_name_field.value:
            page.snack_bar = ft.SnackBar(
                content=ft.Text("Instructor name cannot be blank!")
            )
            page.snack_bar.open = True
            page.update()
        elif instructor_name_field.value:
            instructor = mr.create_intructor(instructor_name_field.value)
            instructor_obj.append(instructor)

            print(instructor_obj)

            delete_button = ft.IconButton(
                icon=ft.icons.DELETE,
                icon_color="red",
                on_click=lambda e, instructor=instructor: delete_instructor(instructor),
            )

            # Update the instructor list view
            instructor_list_view.controls.append(
                ft.Card(
                    ft.ListTile(
                        title=ft.Text(instructor_name_field.value),
                        trailing=delete_button,
                    )
                )
            )

            # Update the checkboxes after adding a new instructor
            create_subject_view()

            instructor_name_field.value = ""
            page.update()

    # Function to add a room
    def add_room(e):
        def delete_room(room):
            index = room_obj.index(room)

            if index != -1:
                del room_obj[index]
                del room_list_view.controls[index]
                print(room_obj)
                room_list_view.update()
                page.update()

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

            print(room_obj)

            delete_button = ft.IconButton(
                icon=ft.icons.DELETE,
                icon_color="red",
                on_click=lambda e, room=room: delete_room(room),
            )

            # Update the instructor list view
            room_list_view.controls.append(
                ft.Card(
                    ft.ListTile(
                        title=ft.Text(
                            f"{room.get_room()} - {room.get_room_type().name}"
                        ),
                        trailing=delete_button,
                    )
                )
            )

            room_number_field.value = ""
            room_type_dropdown.value = None
            page.update()

    # Function to add a subject
    def add_subject(e):
        def delete_subject(subject):
            index = subject_obj.index(subject)

            if index != -1:
                del subject_obj[index]
                del subject_list_view.controls[index]
                print(subject_obj)
                subject_list_view.update()
                create_dept_view()
                page.update()

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

        print(subject_obj)

        delete_button = ft.IconButton(
            icon=ft.icons.DELETE,
            icon_color="red",
            on_click=lambda e, subject=subject: delete_subject(subject),
        )

        # update dept view when a subject is added
        create_dept_view()

        # Add the created subject to the subject list view
        subject_list_view.controls.append(
            ft.Card(
                ft.ListTile(
                    title=ft.Text(
                        f"{subject_name_field.value} - {[instructor.get_name() for instructor in subject.get_subject_instructors()]} - {subject._subject_type.name}"
                    ),
                    trailing=delete_button,
                )
            )
        )

        # Reset input fields
        subject_name_field.value = ""
        subject_type_dropdown.value = None
        for checkbox in instructor_checkboxes:
            checkbox.value = False
        page.update()

    def add_dept(e):
        def delete_dept(dept):
            index = dept_obj.index(dept)

            if index != -1:
                del dept_obj[index]
                del dept_list_view.controls[index]
                print(dept_obj)
                dept_list_view.update()

                department_dropdown.options = [
                    ft.dropdown.Option(
                        str(i),
                        text=f"{dept.get_prefix()} - {dept.get_year_level().name}",
                    )
                    for i, dept in enumerate(dept_obj)
                ]

                department_dropdown.update()
                page.update()

        if not course_dropdown.value:
            page.snack_bar = ft.SnackBar(content=ft.Text("Please select a course!"))
            page.snack_bar.open = True
            page.update()
            return
        if not year_level_dropdown.value:
            page.snack_bar = ft.SnackBar(content=ft.Text("Please select a year level!"))
            page.snack_bar.open = True
            page.update()
            return

        # Gather selected instructors
        selected_subjects = [
            subject_obj[i]
            for i, checkbox in enumerate(subject_checkboxes)
            if checkbox.value
        ]

        if not selected_subjects:
            page.snack_bar = ft.SnackBar(
                content=ft.Text("Please select at least one subject!")
            )
            page.snack_bar.open = True
            page.update()
            return

        prefix = course_dropdown.value
        year_level = getattr(mr.YearLevel, year_level_dropdown.value)

        dept = mr.create_dept(prefix, year_level, selected_subjects)

        print(year_level.name)
        dept_obj.append(dept)

        print(dept_obj)

        delete_button = ft.IconButton(
            icon=ft.icons.DELETE,
            icon_color="red",
            on_click=lambda e, dept=dept: delete_dept(dept),
        )

        dept_list_view.controls.append(
            ft.Card(
                ft.ListTile(
                    title=ft.Text(
                        f"{dept.get_prefix()} - {year_level.name} - {[subject.get_subject_name() for subject in selected_subjects]}"
                    ),
                    trailing=delete_button,
                )
            )
        )

        department_dropdown.options = [
            ft.dropdown.Option(
                str(i), text=f"{dept.get_prefix()} - {dept.get_year_level().name}"
            )
            for i, dept in enumerate(dept_obj)
        ]

        # Reset input fields
        course_dropdown.value = None
        year_level_dropdown.value = None
        for checkbox in subject_checkboxes:
            checkbox.value = False
        page.update()

    # Function to add a block
    def add_block(e):
        def delete_block(block):
            index = block_obj.index(block)

            if index != -1:
                del block_obj[index]
                del block_list_view.controls[index]
                print(block_obj)
                block_list_view.update()
                page.update()

        if not block_number_field.value:
            page.snack_bar = ft.SnackBar(
                content=ft.Text("Block number cannot be blank!")
            )
            page.snack_bar.open = True
            page.update()
            return
        if not department_dropdown.value:
            page.snack_bar = ft.SnackBar(content=ft.Text("Please select a department!"))
            page.snack_bar.open = True
            page.update()
            return

        block_number = block_number_field.value
        selected_index = department_dropdown.value

        # Get the selected department object using the index
        selected_dept = dept_obj[int(selected_index)]

        # Create block and append it to the list
        block = mr.create_block(
            block_number, selected_dept, selected_dept.get_year_level()
        )
        block_obj.append(block)

        print(block_obj)

        delete_button = ft.IconButton(
            icon=ft.icons.DELETE,
            icon_color="red",
            on_click=lambda e, block=block: delete_block(block),
        )
        # Display the added block
        block_list_view.controls.append(
            ft.Card(
                ft.ListTile(
                    title=ft.Text(
                        f"Block {block.get_block_num()} - {selected_dept.get_prefix()} Year {selected_dept.get_year_level()}"
                    ),
                    trailing=delete_button,
                )
            )
        )

        # Reset fields
        block_number_field.value = ""
        department_dropdown.value = None
        year_level_dropdown.value = None
        page.update()

    def generate_schedule(e):
        # when generate schedule is pressed again, clear it
        if len(images_container.controls) != 0:
            images_container.controls.clear()

        # Make the loading screen visible on press
        loading_indicator.visible = True
        page.update()

        try:
            # Ensure the fields are not None and convert to int
            population_size = (
                int(population_size_field.value) if population_size_field.value else 0
            )
            subpopulation_size = (
                int(subpopulation_size_field.value)
                if subpopulation_size_field.value
                else 0
            )
            migration_interval = (
                int(migration_interval_field.value)
                if migration_interval_field.value
                else 0
            )
            generations = int(generation_field.value) if generation_field.value else 0

            # Ensure the slider values are not None and convert to float
            migration_rate = (
                float(migration_rate_slider.value)
                if migration_rate_slider.value
                else 0.1
            )
            elitism_rate = (
                float(elitism_rate_slider.value) if elitism_rate_slider.value else 0.1
            )

            # Pass the values to your ImprovedGeneticAlgorithm class
            ga = mr.ImprovedGeneticAlgorithm(
                population_size=population_size,
                blocks=block_obj,
                rooms=room_obj,
                fitness_limit=1,
                num_subpopulations=subpopulation_size,
                migration_interval=migration_interval,
                migration_rate=migration_rate,
                elitism_rate=elitism_rate,
            )

            evolution_history, elapsed_time, best_generation = ga.evolve(
                generations=generations
            )
            print(evolution_history)
            print(elapsed_time)
            print(best_generation)
            best_schedule = evolution_history[-1][1]  # Get the best schedule

            for idx, block in enumerate(block_obj):
                # best_schedule.print_block_schedule(block)
                best_schedule.generate_visual_schedule(
                    block, f"block{idx+1}_schedule.png"
                )

            # Check if the images are ready and hide loading widget
            images = check_images_available(len(block_obj))
            loading_indicator.visible = False

            if images:
                # Update UI to show images
                images_container.controls.clear()
                for img in images:
                    images_container.controls.append(
                        ft.Image(src=img, width=400, height=300)
                    )
                page.update()
            else:
                page.snack_bar = ft.SnackBar(content=ft.Text("No images found!"))
                page.snack_bar.open = True
                page.update()

        except ValueError:
            page.snack_bar = ft.SnackBar(
                content=ft.Text("Please enter valid numeric values.")
            )
            page.snack_bar.open = True
            page.update()

    # Views for creating subjects and instructors
    create_subjects_view = ft.View(
        "/create_subjects",
        controls=[
            ft.SafeArea(
                ft.Column(
                    controls=[
                        ft.Text(
                            "SchedTool",
                            style=ft.TextStyle(size=24, weight=ft.FontWeight.BOLD),
                        ),
                        ft.Text(
                            "Step 2 of 6",
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
                            ]
                        ),
                    ]
                ),
                minimum_padding=32,
            )
        ],
    )

    create_subjects_view.scroll = ft.ScrollMode.ALWAYS

    # Views for rooms
    create_room_view = ft.View(
        "/create_room",
        controls=[
            ft.SafeArea(
                ft.Column(
                    controls=[
                        ft.Text(
                            "SchedTool",
                            style=ft.TextStyle(size=24, weight=ft.FontWeight.BOLD),
                        ),
                        ft.Text(
                            "Step 3 of 6",
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
                                ft.ElevatedButton(
                                    text="Previous",
                                    on_click=go_back_to_instructors,
                                ),
                                ft.Container(expand=True, expand_loose=True),
                                ft.ElevatedButton(
                                    text="Next",
                                    on_click=go_to_create_departments,
                                ),
                            ]
                        ),
                    ]
                ),
                minimum_padding=32,
            )
        ],
    )
    create_room_view.scroll = ft.ScrollMode.ALWAYS

    # Views for department
    create_depts_view = ft.View(
        "/create_dept",
        controls=[
            ft.SafeArea(
                ft.Column(
                    controls=[
                        ft.Text(
                            "SchedTool",
                            style=ft.TextStyle(size=24, weight=ft.FontWeight.BOLD),
                        ),
                        ft.Text(
                            "Step 4 of 6",
                            style=ft.TextStyle(color=ft.colors.GREY),
                        ),
                        ft.Container(height=16),
                        ft.Text(
                            "Create Department",
                            style=ft.TextStyle(weight=ft.FontWeight.BOLD),
                        ),
                        ft.Container(height=8),
                        course_dropdown,
                        ft.Container(height=8),
                        year_level_dropdown,
                        ft.Container(height=8),
                        ft.Text("Select Subjects:"),
                        create_dept_view(),  # This will dynamically generate the checkboxes
                        ft.Row(
                            controls=[
                                ft.ElevatedButton(
                                    text="Add Department",
                                    on_click=add_dept,
                                ),
                            ]
                        ),
                        ft.Container(height=8),
                        ft.Text("Added Departments:"),
                        dept_list_view,
                        ft.Row(
                            controls=[
                                ft.ElevatedButton(
                                    text="Previous",
                                    on_click=go_back_to_instructors,
                                ),
                                ft.Container(expand=True, expand_loose=True),
                                ft.ElevatedButton(
                                    text="Next",
                                    on_click=go_to_create_block,
                                ),
                            ]
                        ),
                    ]
                ),
                minimum_padding=32,
            )
        ],
    )
    create_depts_view.scroll = ft.ScrollMode.ALWAYS

    # Views for block
    create_block_view = ft.View(
        "/create_block",
        controls=[
            ft.SafeArea(
                ft.Column(
                    controls=[
                        ft.Text(
                            "SchedTool",
                            style=ft.TextStyle(size=24, weight=ft.FontWeight.BOLD),
                        ),
                        ft.Text(
                            "Step 5 of 6",
                            style=ft.TextStyle(color=ft.colors.GREY),
                        ),
                        ft.Container(height=16),
                        ft.Text(
                            "Create Block",
                            style=ft.TextStyle(weight=ft.FontWeight.BOLD),
                        ),
                        ft.Container(height=8),
                        block_number_field,
                        ft.Container(height=8),
                        department_dropdown,
                        ft.Container(height=8),
                        ft.Row(
                            controls=[
                                ft.ElevatedButton(
                                    text="Add Block",
                                    on_click=add_block,
                                    width=120,
                                    height=40,
                                ),
                            ]
                        ),
                        ft.Container(height=8),
                        ft.Text("Added Blocks:"),
                        block_list_view,
                        ft.Container(height=8),
                        ft.Row(
                            controls=[
                                ft.ElevatedButton(
                                    text="Previous",
                                    on_click=go_back_to_instructors,
                                ),
                                ft.Container(expand=True, expand_loose=True),
                                ft.ElevatedButton(
                                    text="Next",
                                    on_click=go_to_create_config,
                                ),
                            ]
                        ),
                    ]
                ),
                minimum_padding=32,
            )
        ],
    )
    create_block_view.scroll = ft.ScrollMode.ALWAYS

    # Views for config screen
    create_config_view = ft.View(
        "/create_config",
        controls=[
            ft.SafeArea(
                ft.Column(
                    controls=[
                        ft.Text(
                            "SchedTool",
                            style=ft.TextStyle(size=24, weight=ft.FontWeight.BOLD),
                        ),
                        ft.Text(
                            "Step 6 of 6",
                            style=ft.TextStyle(color=ft.colors.GREY),
                        ),
                        ft.Container(height=16),
                        ft.Text(
                            "Create Config",
                            style=ft.TextStyle(weight=ft.FontWeight.BOLD),
                        ),
                        ft.Container(height=8),
                        population_size_field,
                        ft.Container(height=8),
                        subpopulation_size_field,
                        ft.Container(height=8),
                        migration_interval_field,
                        ft.Container(height=8),
                        migration_rate_text,
                        ft.Container(height=4),
                        migration_rate_slider,
                        ft.Container(height=8),
                        elitism_rate_text,
                        ft.Container(height=4),
                        elitism_rate_slider,
                        ft.Container(height=8),
                        generation_field,
                        ft.Container(height=8),
                        ft.Row(
                            controls=[
                                ft.ElevatedButton(
                                    text="Generate Schedule",
                                    on_click=generate_schedule,
                                    width=120,
                                    height=40,
                                ),
                            ]
                        ),
                        ft.Container(height=8),
                        # Loading Indicator
                        ft.Row(
                            alignment=ft.MainAxisAlignment.CENTER,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            controls=[loading_indicator],
                        ),
                        # Image display container
                        ft.Row(
                            alignment=ft.MainAxisAlignment.CENTER,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            controls=[images_container],
                        ),
                        ft.Container(height=8),
                        ft.Row(
                            controls=[
                                ft.ElevatedButton(
                                    text="Previous",
                                    on_click=go_back_to_instructors,
                                ),
                                ft.Container(expand=True, expand_loose=True),
                            ]
                        ),
                    ]
                ),
                minimum_padding=32,
            )
        ],
    )
    create_config_view.scroll = ft.ScrollMode.ALWAYS

    page.scroll = ft.ScrollMode.ALWAYS

    # Main page to add instructors
    page.add(
        ft.SafeArea(
            ft.Column(
                controls=[
                    ft.Text(
                        "SchedTool",
                        style=ft.TextStyle(size=24, weight=ft.FontWeight.BOLD),
                    ),
                    ft.Text(
                        "Step 1 of 6",
                        style=ft.TextStyle(color=ft.colors.GREY),
                    ),
                    ft.Container(height=16),
                    ft.Text(
                        "Create Instructors",
                        style=ft.TextStyle(weight=ft.FontWeight.BOLD),
                    ),
                    ft.Container(height=8),
                    instructor_name_field,
                    ft.Container(height=8),
                    ft.Row(
                        controls=[
                            ft.ElevatedButton(
                                text="Add Instructor",
                                on_click=add_instructor,
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
                            ),
                        ]
                    ),
                ],
            ),
            minimum_padding=32,
        )
    )


ft.app(main)
