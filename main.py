# '''
# Class Scheduler
# -- The goal is to use a genetic algorithm to create a class schedule that has no conflicts

# CONSTRAINTS
# These are the constraints to prevent conflicts from schedules
#  - A professor cannot be in two rooms at once in a given time slot
#  - A room can only accomodate one block per time slot.
#  - A block can only take one subject per time slot
#  - Each subject has their own length in hours which is the time they would occupy a room
#  - The available time is only 7am to 9pm.
#  - A subject can occupy any time slot as long as it doesn't conflict with another subject,
#    it fulfills the subject's length in hours, and is within 7am - 9pm

# Additional Constraints
#  - The available days in a week is Monday to Friday
#  - A subject may only be plotted twice in a single week but not in the same day
#  - After each subject, there should be an interval of between 5-10 minutes
#  - After every 5 hours, there should be a break lasting a range of 30 minutes to 1 hour.
#  - A room must be defined by its type (e.g, Lab, Lecture)
#  - A subject must define what type of room it will use (e.g, Lab, Lecture or both)
#  - If a subject has both Lecture and Lab components, it should use a Lab room in the first
#    meeting in a week, and Lecture room in the second week or vice versa.
#  - A subject should be able to occupy decimal hours (e.g., 1 hour and 30 minutes, etc)
#  - A subject can have multiple available instructors and it randomly selects from the list of instructors.
# '''
import math
import random
import time
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from itertools import combinations
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont
from tabulate import tabulate

#  TODO:
# [ ] - implement room types
# [ ] - implement additional time constraints
# [ ] - print the table wherein each class would have their own printed schedule


def generate_visual_schedule(schedule, block, filename="class_schedule.png"):
    # Set up the image
    width, height = 1200, 900
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)

    # Try to load Arial font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 12)
        title_font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    # Define colors
    colors = ["#FFA07A", "#98FB98", "#87CEFA", "#DDA0DD", "#F0E68C"]

    # Draw title
    title = f"Class Schedule for {block.get_block_dept().get_dept_prefix()}-{block.get_block_number()}"
    draw.text((20, 20), title, font=title_font, fill="black")

    # Define grid
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    times = [f"{h:02d}:00" for h in range(7, 22)]  # 7 AM to 9 PM
    cell_width = (width - 100) // len(days)
    cell_height = (height - 100) // (
        len(times) * 2
    )  # Divide each hour into two 30-minute slots

    # Draw grid
    for i, day in enumerate(days):
        draw.text((100 + i * cell_width + 5, 60), day, font=font, fill="black")
        for j, t in enumerate(times):
            draw.text((20, 100 + j * cell_height * 2), t, font=font, fill="black")
            draw.line(
                [(100, 80 + j * cell_height * 2), (width, 80 + j * cell_height * 2)],
                fill="black",
            )

    for i in range(len(days) + 1):
        draw.line(
            [(100 + i * cell_width, 80), (100 + i * cell_width, height)], fill="black"
        )

    # Plot classes
    for day in schedule:
        day_index = days.index(day.name.capitalize())
        for start_time, end_time, subject, room in schedule[day]:
            start_minutes = start_time.hour * 60 + start_time.minute - 7 * 60
            end_minutes = end_time.hour * 60 + end_time.minute - 7 * 60
            if end_minutes <= start_minutes:  # Handle classes ending after midnight
                end_minutes = 14 * 60  # Set to 9 PM

            start_y = 80 + (start_minutes * cell_height) // 30
            end_y = 80 + (end_minutes * cell_height) // 30

            color = random.choice(colors)
            draw.rectangle(
                [
                    100 + day_index * cell_width,
                    start_y,
                    100 + (day_index + 1) * cell_width,
                    end_y,
                ],
                fill=color,
                outline="black",
            )

            text = f"{subject.get_subject_name()}\n{room.get_room_num()}\n{start_time.strftime('%I:%M %p')}-{end_time.strftime('%I:%M %p')}"
            draw.text(
                (105 + day_index * cell_width, start_y + 5),
                text,
                font=font,
                fill="black",
            )

    # Save the image
    image.save(filename)
    print(f"Schedule image saved as {filename}")


class RoomType(Enum):
    LECTURE = 1
    LAB = 2


class Day(IntEnum):
    MONDAY = 1
    TUESDAY = 2
    WEDNESDAY = 3
    THURSDAY = 4
    FRIDAY = 5


class Instructor:
    def __init__(self, name: str) -> None:
        self._name = name

    def get_name(self) -> str:
        return self._name

    def __str__(self) -> str:
        return self._name


class Room:
    def __init__(self, room_num: str, max_capacity: int, room_type: RoomType) -> None:
        self._room_num = room_num
        self._max_capacity = max_capacity
        self._room_type = room_type

    def get_room_num(self) -> str:
        return self._room_num

    def get_max_capacity(self) -> int:
        return self._max_capacity

    def get_room_type(self) -> RoomType:
        return self._room_type


# TODO: This should support minutes as well instead of full hours
class Subject:
    def __init__(
        self, name: str, instructors: list[Instructor], duration: timedelta
    ) -> None:
        self._name = name
        self._instructors = instructors
        self.duration = duration
        self._instructor = None
        self.set_subject_instructor()

    def set_subject_instructor(self) -> None:
        self._instructor = random.choice(self._instructors)

    def get_subject_name(self) -> str:
        return self._name

    def get_subject_instructors(self) -> list[Instructor]:
        return self._instructors

    def get_subject_duration(self) -> timedelta:
        return self.duration

    def __str__(self) -> str:
        return self._name


class Department:
    def __init__(self, prefix: str, subjects: list[Subject]) -> None:
        self._prefix = prefix  # CS, WD, NA, EMC, CYB
        self._subjects = subjects

    def get_dept_prefix(self) -> str:
        return self._prefix

    def get_dept_subjects(self) -> list[Subject]:
        return self._subjects


class Block:
    def __init__(
        self, number: str, dept: Department, subjects: list[Subject], num_students: int
    ) -> None:
        self._number = number  # 303, 102, 202
        self._dept = dept  # dept.prefix i.e., CS,
        self._subjects = subjects
        self._num_students = num_students
        self._room = None
        self._instructor = None

    def get_block_number(self) -> str:
        return self._number

    def get_block_dept(self) -> Department:
        return self._dept

    def get_block_subjects(self) -> list[Subject]:
        return self._subjects

    def get_room(self):
        return self._room

    def set_room(self, room):
        self._room = room

    def set_instructor(self, instructor):
        self._instructor = instructor

    def __str__(self) -> str:
        return f"${self._dept.get_dept_prefix()}-${self._number}, ${self._subjects}, ${self._room.get_room_num()}, ${self._instructor.get_name()}"  # type: ignore


class TimeSlot:
    def __init__(self, day: Day, start_time: datetime, end_time: datetime):
        self._day = day
        self.start_time = start_time
        self.end_time = end_time

    def get_day(self) -> Day:
        return self._day

    def __str__(self):
        return f"{self.start_time.strftime('%I:%M %p')} - {self.end_time.strftime('%I:%M %p')}"


class Schedule:
    def __init__(self, blocks: List[Block], rooms: List[Room]):
        self.blocks = blocks
        self.rooms = rooms
        self.assignments = []  # List of (Block, Subject, Room, TimeSlot) tuples
        self.generate_random_schedule()

    def generate_random_schedule(self):
        self.assignments = []
        for block in self.blocks:
            for subject in block.get_block_subjects():
                # Schedule the subject twice
                scheduled_days = set()
                for _ in range(2):
                    room = random.choice(self.rooms)

                    # Generate a random start time between 7:00 AM and 9:00 PM, aligned to 5-minute intervals
                    start_time = datetime.combine(
                        datetime.today(), datetime.min.time()
                    ) + timedelta(
                        hours=7,
                        minutes=random.randint(0, 14 * 12)
                        * 5,  # 14 hours * 12 5-minute intervals
                    )

                    end_time = start_time + subject.get_subject_duration()

                    # If end time is after 9:00 PM, adjust start time
                    if end_time.hour >= 21:
                        start_time = (
                            datetime.combine(datetime.today(), datetime.min.time())
                            + timedelta(hours=21)
                            - subject.get_subject_duration()
                        )
                        end_time = datetime.combine(
                            datetime.today(), datetime.min.time()
                        ) + timedelta(hours=21)

                    # Choose a day that hasn't been scheduled yet
                    available_days = set(Day) - scheduled_days
                    if available_days:
                        day = random.choice(list(available_days))
                        scheduled_days.add(day)
                    else:
                        day = random.choice(list(Day))  # Fallback, should rarely happen

                    time_slot = TimeSlot(day, start_time, end_time)
                    self.assignments.append((block, subject, room, time_slot))
        # print(f"The assignments are: {len(self.assignments)}")

    def calculate_fitness(self) -> float:
        conflicts = defaultdict(int)
        total_assignments = len(self.assignments)
        subject_occurrences = defaultdict(list)

        def time_overlap(t1, t2):
            return t1.start_time < t2.end_time and t2.start_time < t1.end_time

        def time_diff_minutes(t1, t2):
            return abs(int((t1 - t2).total_seconds() / 60))

        # Pre-process assignments
        day_block_assignments = defaultdict(list)
        day_assignments = defaultdict(list)

        for block, subject, room, t in self.assignments:
            day = t.get_day()
            day_block_assignments[(day, block)].append((subject, room, t))

            conflicts["time"] += t.start_time.hour < 7 or t.end_time.hour > 21
            subject_occurrences[(block, subject)].append(day)

        # Check for conflicts
        for (day, block), assignments in day_block_assignments.items():
            sorted_assignments = sorted(assignments, key=lambda x: x[2].start_time)

            for (subject1, room1, time1), (subject2, room2, time2) in combinations(
                sorted_assignments, 2
            ):
                if time_overlap(time1, time2):
                    conflicts["time"] += 1
                    conflicts["room"] += room1 == room2
                    conflicts["instructor"] += bool(
                        set(subject1.get_subject_instructors())
                        & set(subject2.get_subject_instructors())
                    )
                    conflicts["block"] += 1
                    if subject1._instructor == subject2._instructor:
                        conflicts["instructor"] += 1
                interval = time_diff_minutes(time2.start_time, time1.end_time)
                if interval < 5:
                    conflicts["interval"] += (
                        1  # Penalize if interval is less than 5 minutes
                    )
                elif interval > 10 and interval % 5 != 0:
                    conflicts["interval"] += (
                        1  # Penalize if interval exceeds 10 and is not divisible by 5
                    )

            # Check for breaks after every 5 hours
            cumulative_time = timedelta()
            last_break_end = sorted_assignments[0][2].start_time
            for _, _, time_slot in sorted_assignments:
                cumulative_time += time_slot.end_time - time_slot.start_time
                if cumulative_time >= timedelta(hours=5):
                    break_duration = time_diff_minutes(
                        time_slot.end_time, last_break_end
                    )
                    conflicts["time"] += break_duration < 30 or break_duration > 60
                    cumulative_time = timedelta()
                    last_break_end = time_slot.end_time

        # For cases where block1 and block2 have overlapping or similar schedule
        for day, assignments in day_assignments.items():
            sorted_assignments = sorted(
                assignments, key=lambda x: x[3].start_time
            )  # Sort by time

            for (block1, subject1, room1, time1), (
                block2,
                subject2,
                room2,
                time2,
            ) in combinations(sorted_assignments, 2):
                # If block1 and block 2 are different blocks
                # and the time overlaps, penalize the schedule
                if block1 != block2 and time_overlap(time1, time2):
                    # If block1 and block2 have overlapping schedules,
                    # penalize if they have the same room, same instructors
                    if room1 == room2:
                        conflicts["room"] += 1
                        conflicts["time"] += 1

                    # conflicts["instructor"] += bool(
                    #     set(subject1.get_subject_instructors())
                    #     & set(subject2.get_subject_instructors())
                    # )

                    # Check for same subject in different blocks
                    if subject1 == subject2:
                        if subject1._instructor == subject2._instructor:
                            conflicts["instructor"] += 1

        # Check subject occurrences
        conflicts["subject_occurrence"] = sum(
            len(occurrences) != 2 or len(set(occurrences)) != 2
            for occurrences in subject_occurrences.values()
        )

        # Calculate total conflicts with weights
        weights = {
            "time": 20,
            "room": 3,
            "instructor": 3,
            "block": 3,
            "interval": 1,
            "subject_occurrence": 2,
        }
        total_conflicts = sum(conflicts[k] * weights[k] for k in weights)

        # Normalize conflicts and calculate fitness
        max_possible_conflicts = total_assignments * sum(weights.values())

        return 1 - (total_conflicts / max_possible_conflicts)

    def print_block_schedule(self, block: Block):
        schedule = {day: [] for day in Day}

        for b, subject, room, time_slot in self.assignments:
            if b == block:
                day = time_slot.get_day()
                start_time = time_slot.start_time
                end_time = time_slot.end_time
                schedule[day].append((start_time, end_time, subject, room))

        # Sort schedules for each day
        for day in Day:
            schedule[day].sort(key=lambda x: x[0])

        # Generate all possible time slots
        all_time_slots = []
        current_time = datetime.combine(
            datetime.today(), datetime.min.time()
        ) + timedelta(hours=7)
        end_of_day = datetime.combine(
            datetime.today(), datetime.min.time()
        ) + timedelta(hours=21)

        while current_time < end_of_day:
            all_time_slots.append(current_time)
            current_time += timedelta(minutes=5)

        table_data = []
        for time_slot in all_time_slots:
            row = [time_slot.strftime("%I:%M %p")]
            for day in Day:
                cell_content = ""
                for start_time, end_time, subject, room in schedule[day]:
                    if start_time <= time_slot < end_time:
                        cell_content = (
                            f"{subject.get_subject_name()}\n{room.get_room_num()}"
                        )
                        break
                row.append(cell_content)
            table_data.append(row)

        # Remove consecutive duplicate rows
        compressed_table_data = []
        for row in table_data:
            if not compressed_table_data or row != compressed_table_data[-1]:
                compressed_table_data.append(row)

        # Print the table
        headers = ["Time"] + [day.name.capitalize() for day in Day]
        print(
            f"\nSchedule for {block.get_block_dept().get_dept_prefix()}-{block.get_block_number()}:"
        )
        print(tabulate(compressed_table_data, headers=headers, tablefmt="grid"))

        # Print intervals between classes
        for day in Day:
            if schedule[day]:
                print(f"\nIntervals for {day.name}:")
                for i in range(len(schedule[day]) - 1):
                    current_end = schedule[day][i][1]
                    next_start = schedule[day][i + 1][0]
                    interval = (next_start - current_end).total_seconds() / 60
                    print(
                        f"  {current_end.strftime('%I:%M %p')} - {next_start.strftime('%I:%M %p')}: {interval} minutes"
                    )

    def generate_visual_schedule(self, block, filename="class_schedule.png"):
        schedule = {day: [] for day in Day}

        for b, subject, room, time_slot in self.assignments:
            if b == block:
                day = time_slot.get_day()
                start_time = time_slot.start_time
                end_time = time_slot.end_time
                schedule[day].append((start_time, end_time, subject, room))

        # Sort schedules for each day
        for day in Day:
            schedule[day].sort(key=lambda x: x[0])

        generate_visual_schedule(schedule, block, filename)


class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int,
        mutation_rate: float,
        blocks: List[Block],
        rooms: List[Room],
        fitness_limit=1.00,
        elitism_rate=0.1,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.blocks = blocks
        self.rooms = rooms
        self.population = [Schedule(blocks, rooms) for _ in range(population_size)]
        self.fitness_limit = fitness_limit
        self.elitism_rate = elitism_rate

    def _select_parent(self) -> Schedule:
        # Arbitrary precision level
        precision_level = 0.5

        # Yamane's Formula
        sample_size = self.population_size / (
            1 + (self.population_size * (precision_level**2))
        )

        # select a sample from the given population and calculated sample size
        tournament = random.sample(self.population, math.floor(sample_size))

        # select the fittest (highest/max) schedule to be the parent based on the
        # tournament list and their fitness score
        return max(tournament, key=lambda schedule: schedule.calculate_fitness())

    def _crossover(self, parent1: Schedule, parent2: Schedule) -> Schedule:
        child = Schedule(self.blocks, self.rooms)

        midpoint = len(parent1.assignments) // 2

        child.assignments = (
            parent1.assignments[:midpoint] + parent2.assignments[midpoint:]
        )

        return child

    def _mutate(self, schedule: Schedule):
        for i in range(len(schedule.assignments)):
            if random.random() < self.mutation_rate:
                block, subject, _, _ = schedule.assignments[i]
                room = random.choice(self.rooms)

                # Generate a random start time between 7:00 AM and 9:00 PM, aligned to 5-minute intervals
                start_time = datetime.combine(
                    datetime.today(), datetime.min.time()
                ) + timedelta(
                    hours=7,
                    minutes=random.randint(0, 14 * 12)
                    * 5,  # 14 hours * 12 5-minute intervals
                )

                end_time = start_time + subject.get_subject_duration()

                # If end time is after 9:00 PM, adjust start time
                if end_time.hour >= 21:
                    start_time = (
                        datetime.combine(datetime.today(), datetime.min.time())
                        + timedelta(hours=21)
                        - subject.get_subject_duration()
                    )
                    end_time = datetime.combine(
                        datetime.today(), datetime.min.time()
                    ) + timedelta(hours=21)

                # Ensure the new day is different from the other occurrence of this subject
                current_days = [
                    assign[3].get_day()
                    for assign in schedule.assignments
                    if assign[0] == block and assign[1] == subject
                ]

                available_days = list(set(Day) - set(current_days))
                if available_days:
                    day = random.choice(available_days)
                else:
                    day = random.choice(list(Day))

                time_slot = TimeSlot(day, start_time, end_time)
                schedule.assignments[i] = (block, subject, room, time_slot)

    def evolve(self, generations: int):
        evolution_history = []
        start_time = time.time()
        for gen in range(generations):
            self.population.sort(key=lambda x: x.calculate_fitness(), reverse=True)

            # Elitism
            elite_size = int(self.population_size * self.elitism_rate)
            new_population = self.population[:elite_size]

            while len(new_population) < self.population_size:
                parent1 = self._select_parent()
                parent2 = self._select_parent()
                child = self._crossover(parent1, parent2)
                self._mutate(child)
                new_population.append(child)

            self.population = new_population

            best_schedule = self.get_best_schedule()
            evolution_history.append((gen, best_schedule))

            if math.isclose(
                best_schedule.calculate_fitness(), self.fitness_limit, abs_tol=0.001
            ):
                break

        elapsed_time = time.time() - start_time
        best_generation = evolution_history[-1][0]
        return evolution_history, elapsed_time, best_generation

    def get_best_schedule(self) -> Schedule:
        return max(self.population, key=lambda schedule: schedule.calculate_fitness())


class ImprovedGeneticAlgorithm:
    def __init__(
        self,
        population_size: int,
        blocks: List[Block],
        rooms: List[Room],
        fitness_limit=1.00,
        num_subpopulations=5,
        migration_interval=10,
        migration_rate=0.1,
        elitism_rate=0.1,
    ):
        self.population_size = population_size
        self.blocks = blocks
        self.rooms = rooms
        self.fitness_limit = fitness_limit
        self.num_subpopulations = num_subpopulations
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self.elitism_rate = elitism_rate

        # Initialize subpopulations
        subpopulation_size = population_size // num_subpopulations
        self.subpopulations = [
            [Schedule(blocks, rooms) for _ in range(subpopulation_size)]
            for _ in range(num_subpopulations)
        ]

        # Assign distinct mutation rates to each subpopulation
        self.subpopulation_mutation_rates = [
            random.uniform(0.01, 0.05) for _ in range(num_subpopulations)
        ]

    def _select_parent(self, subpopulation: List[Schedule]) -> Schedule:
        tournament_size = max(2, len(subpopulation) // 5)
        tournament = random.sample(subpopulation, tournament_size)
        return max(tournament, key=lambda schedule: schedule.calculate_fitness())

    def _crossover(
        self, parent1: Schedule, parent2: Schedule, strategy: str
    ) -> Schedule:
        child = Schedule(self.blocks, self.rooms)
        if strategy == "single_point":
            # Single-point crossover
            midpoint = len(parent1.assignments) // 2
            child.assignments = (
                parent1.assignments[:midpoint] + parent2.assignments[midpoint:]
            )
        elif strategy == "multi_parent":
            # Multi-parent crossover (example using 3 parents)
            parent3 = random.choice(
                self.subpopulations[0]
            )  # Example of selecting an additional parent
            third_point = len(parent1.assignments) // 3
            child.assignments = (
                parent1.assignments[:third_point]
                + parent2.assignments[third_point : 2 * third_point]
                + parent3.assignments[2 * third_point :]
            )
        return child

    def _mutate(self, schedule: Schedule, mutation_rate: float):
        for i in range(len(schedule.assignments)):
            if random.random() < mutation_rate:
                block, subject, _, _ = schedule.assignments[i]
                room = random.choice(self.rooms)
                start_time = datetime.combine(
                    datetime.today(), datetime.min.time()
                ) + timedelta(hours=7, minutes=random.randint(0, 14 * 12) * 5)
                end_time = start_time + subject.get_subject_duration()
                if end_time.hour >= 21:
                    start_time = (
                        datetime.combine(datetime.today(), datetime.min.time())
                        + timedelta(hours=21)
                        - subject.get_subject_duration()
                    )
                    end_time = datetime.combine(
                        datetime.today(), datetime.min.time()
                    ) + timedelta(hours=21)
                current_days = [
                    assign[3].get_day()
                    for assign in schedule.assignments
                    if assign[0] == block and assign[1] == subject
                ]
                available_days = list(set(Day) - set(current_days))
                day = (
                    random.choice(available_days)
                    if available_days
                    else random.choice(list(Day))
                )
                time_slot = TimeSlot(day, start_time, end_time)
                schedule.assignments[i] = (block, subject, room, time_slot)

    def _migrate(self):
        for i in range(self.num_subpopulations):
            next_subpop = (i + 1) % self.num_subpopulations
            migrants = random.sample(
                self.subpopulations[i],
                int(len(self.subpopulations[i]) * self.migration_rate),
            )
            self.subpopulations[next_subpop].extend(migrants)
            self.subpopulations[next_subpop] = sorted(
                self.subpopulations[next_subpop],
                key=lambda x: x.calculate_fitness(),
                reverse=True,
            )[: len(self.subpopulations[i])]

    def evolve(self, generations: int) -> Tuple[List[Tuple[int, Schedule]], float, int]:
        evolution_history = []
        start_time = time.time()

        for gen in range(generations):
            # Start timer for this generation
            gen_start_time = time.time()

            for subpop_idx, subpopulation in enumerate(self.subpopulations):
                new_subpopulation = []

                # Elitism
                elites = int(len(subpopulation) * self.elitism_rate)
                new_subpopulation.extend(
                    sorted(
                        subpopulation, key=lambda x: x.calculate_fitness(), reverse=True
                    )[:elites]
                )

                # Generate new individuals
                crossover_strategy = random.choice(
                    ["single_point", "multi_parent"]
                )  # Randomly select crossover strategy
                while len(new_subpopulation) < len(subpopulation):
                    parent1 = self._select_parent(subpopulation)
                    parent2 = self._select_parent(subpopulation)
                    child = self._crossover(parent1, parent2, crossover_strategy)

                    # Use subpopulation-specific mutation rates
                    dynamic_mutation_rate = self.subpopulation_mutation_rates[
                        subpop_idx
                    ]
                    self._mutate(child, dynamic_mutation_rate)
                    new_subpopulation.append(child)

                # Check for premature convergence and trigger disaster event if needed
                if self._detect_premature_convergence(subpopulation):
                    self._trigger_disaster_event(new_subpopulation)

                self.subpopulations[subpop_idx] = new_subpopulation

            # Migration
            if gen % self.migration_interval == 0:
                self._migrate()

            best_schedule = self.get_best_schedule()
            evolution_history.append((gen, best_schedule))

            # Stop if we've found a perfect solution
            if math.isclose(best_schedule.calculate_fitness(), self.fitness_limit):
                break

            # End timer for this generation and print results
            gen_end_time = time.time()
            gen_elapsed_time = gen_end_time - gen_start_time
            print(f"Generation {gen + 1} completed in {gen_elapsed_time:.2f} seconds.")

        # Finalize total evolution time
        total_elapsed_time = time.time() - start_time
        best_generation = evolution_history[-1][0]
        print(f"Evolution completed in {total_elapsed_time:.2f} seconds.")
        return evolution_history, total_elapsed_time, best_generation

    def get_best_schedule(self) -> Schedule:
        return max(
            (schedule for subpop in self.subpopulations for schedule in subpop),
            key=lambda schedule: schedule.calculate_fitness(),
        )

    def _calculate_diversity(self, subpopulation: List[Schedule]) -> float:
        fitnesses = [schedule.calculate_fitness() for schedule in subpopulation]
        avg_fitness = sum(fitnesses) / len(fitnesses)
        variance = sum((f - avg_fitness) ** 2 for f in fitnesses) / len(fitnesses)
        return (
            math.sqrt(variance) / avg_fitness
        )  # Coefficient of variation as a measure of diversity

    def _trigger_disaster_event(
        self, subpopulation: List[Schedule], disaster_rate: float = 0.5
    ):
        disaster_size = int(
            len(subpopulation) * disaster_rate
        )  # Replace a portion of the population
        new_individuals = [
            Schedule(self.blocks, self.rooms) for _ in range(disaster_size)
        ]
        subpopulation[-disaster_size:] = new_individuals

    def _detect_premature_convergence(self, subpopulation: List[Schedule]) -> bool:
        diversity = self._calculate_diversity(subpopulation)
        return diversity < 0.1  # Threshold for detecting premature convergence


if __name__ == "__main__":
    # Create sample data (instructors, subjects, departments, blocks, rooms)
    sir_uly = Instructor("Sir Ulysses Monsale")
    maam_lou = Instructor("Ma'am Louella Salenga")
    sir_glenn = Instructor("Sir Glenn Mañalac")
    sir_lloyd = Instructor("Sir Lloyd Estrada")
    maam_raquel = Instructor("Ma'am Raquel Rivera")
    sir_marc = Instructor("Sir Marc Corporal")
    mam_max = Instructor("Ma'am Max")
    anisa = Instructor("Ma'am Anisa")
    patrick = Instructor("Sir Patrick")
    kyle = Instructor("Sir Kyle")

    IMODSIM = Subject("IMODSIM", [maam_lou, sir_marc], timedelta(hours=2))
    PROBSTAT = Subject("PROBSTAT", [sir_lloyd, mam_max], timedelta(hours=1, minutes=30))
    INTCALC = Subject("INTCALC", [sir_glenn, anisa], timedelta(hours=1, minutes=30))
    ATF = Subject("ATF", [sir_uly, patrick], timedelta(hours=3))
    SOFTENG = Subject("SOFTENG", [maam_raquel, kyle], timedelta(hours=2))

    ROOMS = [
        ["SJH-503", 45, RoomType.LECTURE],
        ["SJH-504", 45, RoomType.LAB],
        ["SJH-505", 45, RoomType.LECTURE],
        ["SJH-506", 45, RoomType.LECTURE],
    ]

    dept = Department("CS", [INTCALC, PROBSTAT, IMODSIM, ATF, SOFTENG])
    dept_subjects = dept.get_dept_subjects()

    block1 = Block("301", dept, dept_subjects, 45)
    block2 = Block("302", dept, dept_subjects, 45)
    block3 = Block("303", dept, dept_subjects, 45)
    block4 = Block("304", dept, dept_subjects, 45)

    rooms = [
        Room(room_num, capacity, room_type) for room_num, capacity, room_type in ROOMS
    ]

    ga = ImprovedGeneticAlgorithm(
        population_size=250,
        blocks=[block1, block2, block3, block4],
        rooms=rooms,
        num_subpopulations=10,
        migration_interval=10,
        migration_rate=0.1,
        elitism_rate=0.1,
    )

    evolution_history, elapsed_time, best_generation = ga.evolve(generations=250)

    print("\nEvolution completed.")
    best_schedule = evolution_history[-1][1]  # Get the best schedule
    for idx, block in enumerate([block1, block2, block3, block4]):
        # best_schedule.print_block_schedule(block)
        best_schedule.generate_visual_schedule(block, f"block{idx+1}_schedule.png")

    best_schedule = ga.get_best_schedule()
    print(f"\nFitness: {best_schedule.calculate_fitness():.4f}")
    print(f"Best solution found in generation: {best_generation}")
    print(f"Time taken to find the best solution: {elapsed_time:.2f} seconds")
