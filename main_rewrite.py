import math
import random
import time as t
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from itertools import combinations

from PIL import Image, ImageDraw, ImageFont

DEBUG = False


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
    title = f"Class Schedule for {block.get_block_dept().get_prefix()}-{block.get_block_num()}"
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
        for j, time in enumerate(times):
            draw.text((20, 100 + j * cell_height * 2), time, font=font, fill="black")
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

            text = f"{subject.get_subject_name()}\n{room.get_room()}\n{start_time.strftime('%I:%M %p')}-{end_time.strftime('%I:%M %p')}"
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


class YearLevel(IntEnum):
    FIRST = 1
    SECOND = 2
    THIRD = 3
    FOURTH = 4


class SubjectType(Enum):
    SPECIALIZED_LECTURE = 1
    SPECIALIZED_LAB = 2
    GENERAL_EDUCATION = 3


class Instructor:
    def __init__(self, name: str) -> None:
        self._name = name

    def get_name(self) -> str:
        return self._name

    def __str__(self) -> str:
        return self._name


class Room:
    def __init__(self, room_num: str, room_type: RoomType) -> None:
        self._room_num = room_num
        self._room_type = room_type

    def get_room(self):
        return self._room_num

    def get_room_type(self) -> RoomType:
        return self._room_type

    def __str__(self) -> str:
        return f"{self._room_num}"


class Subject:
    def __init__(
        self,
        name: str,
        available_instructors: list[Instructor],
        subject_type: SubjectType,
    ) -> None:
        self._name = name
        self._available_instructors = available_instructors
        self._subject_type = subject_type
        self._duration = None
        self._instructor = None
        self.set_subject_duration()
        self.set_subject_instructor()

    def set_subject_instructor(self) -> None:
        self._instructor = random.choice(self._available_instructors)

    def set_subject_duration(self) -> None:
        match self._subject_type:
            case SubjectType.SPECIALIZED_LECTURE:
                self._duration = timedelta(hours=2)
            case SubjectType.SPECIALIZED_LAB:
                self._duration = timedelta(hours=3)
            case SubjectType.GENERAL_EDUCATION:
                self._duration = timedelta(hours=1, minutes=30)

    def get_subject_duration(self) -> timedelta | None:
        return self._duration

    def get_subject_name(self) -> str:
        return self._name

    def get_subject_instructors(self) -> list[Instructor]:
        return self._available_instructors

    def __str__(self) -> str:
        return f"{self._name}"


class Department:
    def __init__(
        self,
        prefix: str,
        year_level_subjects: dict[YearLevel, list[Subject]],
        year_level: YearLevel,
    ) -> None:
        self._prefix = prefix
        self._year_level_subjects = year_level_subjects
        self._year_level = year_level

    def get_prefix(self) -> str:
        return self._prefix

    def get_year_level(self) -> YearLevel:
        return self._year_level

    def get_all_subjects(self) -> dict[YearLevel, list[Subject]]:
        return self._year_level_subjects

    def __str__(self) -> str:
        return f"{self._prefix}"


class Block:
    def __init__(
        self, block_num: str, department: Department, year_level: YearLevel
    ) -> None:
        self._block_num = block_num
        self._department = department
        self._year_level = year_level

    def get_block_dept(self) -> Department:
        return self._department

    def get_block_num(self) -> str:
        return self._block_num

    def get_year_level(self) -> YearLevel:
        return self._year_level

    # get the subjects of a given year level
    def get_level_subjects(self) -> list[Subject]:
        all_subjects = self._department.get_all_subjects()
        return all_subjects[self._year_level]

    def __str__(self) -> str:
        return f"{self._department.get_prefix()}-{self.get_block_num()}"


class TimeSlot:
    def __init__(self, day: Day, start_time: datetime, end_time: datetime) -> None:
        self._day = day
        self._start_time = start_time
        self._end_time = end_time

    def get_day(self) -> Day:
        return self._day

    def __str__(self):
        return f"{self._start_time.strftime('%I:%M %p')} - {self._end_time.strftime('%I:%M %p')}"


class Schedule:
    def __init__(self, blocks: list[Block], rooms: list[Room]):
        self._blocks = blocks
        self._rooms = rooms
        self._assignments = []  # List of (Block, Subject, Room, TimeSlot) tuples
        self.generate_random_schedule()

    def generate_random_time(self, subject: Subject) -> tuple[datetime, datetime]:
        subject_duration = subject.get_subject_duration()
        if subject_duration is None:
            raise ValueError("Subject duration cannot be None")

        # generate a start time. combining today() and min.time() sets the date and time to today midnight
        start_time = datetime.combine(
            datetime.today(), datetime.min.time()
        ) + timedelta(hours=7, minutes=random.randint(0, 14 * 12) * 5)

        end_time = start_time + subject_duration

        # If end time is after 9:00 PM, adjust start time
        if end_time.hour >= 21:
            # Start at 9PM and subtract the subject's duration
            start_time = (
                datetime.combine(datetime.today(), datetime.min.time())
                + timedelta(hours=21)
                - subject_duration
            )
            # adjust the end to be at 9PM
            end_time = datetime.combine(
                datetime.today(), datetime.min.time()
            ) + timedelta(hours=21)

        return start_time, end_time

    def schedule_subject(
        self, block: Block, subject: Subject, day: Day, scheduled_days: set
    ) -> None:
        # Choose a random room
        # TODO: refactor this

        lecture_rooms = list(
            filter(lambda room: room.get_room_type() == RoomType.LECTURE, self._rooms)
        )

        lab_rooms = list(
            filter(lambda room: room.get_room_type() == RoomType.LAB, self._rooms)
        )

        # there should be available lecture rooms
        if (
            lecture_rooms
            and subject._subject_type == SubjectType.GENERAL_EDUCATION
            or subject._subject_type == SubjectType.SPECIALIZED_LECTURE
        ):
            room = random.choice(lab_rooms)

        # there should be available lab rooms
        elif lab_rooms and subject._subject_type == SubjectType.SPECIALIZED_LAB:
            room = random.choice(lecture_rooms)

        # If lab or lecture rooms are empty, just choose a random room from the list of rooms
        # that would severely impact the accuracy of the model however.
        else:
            room = random.choice(self._rooms)

        # generate a random start and end time
        start_time, end_time = self.generate_random_time(subject)

        time_slot = TimeSlot(day, start_time, end_time)
        self._assignments.append((block, subject, room, time_slot))

        # For now the length of the assignments is 5 since it assigns a schedule per day
        scheduled_days.add(day)

    def generate_random_schedule(self) -> None:
        # Make sure assignments are always new
        self._assignments = []

        for block in self._blocks:
            for subject in block.get_level_subjects():
                scheduled_days = set()

                # Schedule the subject twice (or once in the case of Wednesday)
                for _ in range(2):
                    # Choose a day that hasn't been scheduled yet
                    available_days = set(Day) - scheduled_days
                    if available_days:
                        day = random.choice(list(available_days))
                        scheduled_days.add(day)
                    else:
                        # Fallback, should rarely happen
                        day = random.choice(list(Day))

                    # Use the schedule_subject helper function to handle room assignment, time generation, and schedule assignment
                    self.schedule_subject(block, subject, day, scheduled_days)

    def calculate_fitness(self):
        # initialize the weights of each conflict
        conflicts = defaultdict(int)
        total_assignments = len(self._assignments)
        subject_occurrences = defaultdict(list)

        def time_overlap(t1: TimeSlot, t2: TimeSlot) -> bool:
            # the second condition is the more important part in checking if
            # the times do overlap
            return t1._start_time < t2._end_time and t2._start_time < t1._end_time

        def time_diff_minutes(t1: datetime, t2: datetime) -> int:
            return abs(int((t1 - t2).total_seconds() / 60))

        day_block_assignments = defaultdict(list)
        day_assignments = defaultdict(list)

        # unpack the values in self._assignments
        for block, subject, room, time in self._assignments:
            day = time.get_day()
            subject_occurrences[(block, subject)].append(day)

            """
            day_block_assignments = {
                ('Monday', 'Block1'): [(Math, Room101, Time(9:00-10:00)), (Physics, Room102, Time(9:30-10:30))],
                ('Monday', 'Block2'): [(Chemistry, Room103, Time(11:00-12:00))],
                ('Tuesday', 'Block1'): [(English, Room104, Time(9:00-10:00)), (History, Room105, Time(10:00-11:00))],
            }
            """
            # Compare classes that are on the same day and in the same block,
            # which are the only ones that could potentially conflict.
            day_block_assignments[(day, block)].append((subject, room, time))
            day_assignments[day].append((block, subject, room, time))

        for (day, block), assignments in day_block_assignments.items():
            """
                To visualize sorted_assignments better: 
                --- MONDAY, CS-301 ---
                Subject: ATF in SJH-504 from 2024-10-18 15:00:00 to 2024-10-18 17:00:00
                Instructors: ['Sir Uly']

                --- THURSDAY, CS-301 ---
                Subject: ATF in SJH-503 from 2024-10-18 16:45:00 to 2024-10-18 18:45:00
                Instructors: ['Sir Uly']

                --- TUESDAY, CS-301 ---
                Subject: ATF in SJH-504 from 2024-10-18 19:00:00 to 2024-10-18 21:00:00
                Instructors: ['Sir Uly']
            """
            sorted_assignments = sorted(assignments, key=lambda x: x[2]._start_time)

            for (subject1, room1, time1), (subject2, room2, time2) in combinations(
                sorted_assignments, 2
            ):
                if time_overlap(time1, time2):
                    conflicts["time"] += 1

                    # 0 if False, 1 if True
                    conflicts["room"] += room1 == room2

                    # conflicts["instructor"] += bool(
                    #     set(subject1.get_subject_instructors())
                    #     & set(subject2.get_subject_instructors())
                    # )

                    if subject1._instructor == subject2._instructor:
                        conflicts["instructor"] += 1

                interval = time_diff_minutes(time2._start_time, time1._end_time)

                # Penalize if interval is less than 5 minutes
                if interval < 5:
                    conflicts["interval"] += 1

                # Penalize if interval exceeds 10 and is not divisible by 5
                elif interval > 10 and interval % 5 != 0:
                    conflicts["interval"] += 1

            if DEBUG:
                print(f"\n--- {day.name}, {block} ---")
                for subject, room, time in sorted_assignments:
                    print(
                        f"Subject: {subject.get_subject_name()} in {room} from {time._start_time} to {time._end_time}"
                    )
                    print(
                        f"Instructors: {[instr.get_name() for instr in subject.get_subject_instructors()]}"
                    )

            # Check for breaks after every 5 hours
            # cumulative_time = timedelta()
            # last_break_end = sorted_assignments[0][2]._start_time
            # for _, _, time_slot in sorted_assignments:
            #     cumulative_time += time_slot._end_time - time_slot._start_time
            #     if cumulative_time >= timedelta(hours=5):
            #         break_duration = time_diff_minutes(
            #             time_slot._end_time, last_break_end
            #         )
            #         conflicts["time"] += break_duration < 30 or break_duration > 60
            #         cumulative_time = timedelta()
            #         last_break_end = time_slot._end_time

        # For cases where block1 and block2 have overlapping or similar schedule
        for day, assignments in day_assignments.items():
            sorted_assignments = sorted(
                assignments, key=lambda x: x[3]._start_time
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

        weights = {
            "time": 20,
            "room": 3,
            "instructor": 3,
            "block": 3,
            "interval": 1,
        }

        total_conflicts = sum(conflicts[k] * weights[k] for k in weights)
        max_possible_conflicts = total_assignments * sum(weights.values())
        fitness = 1 - (total_conflicts / max_possible_conflicts)
        if DEBUG:
            for (day, block), assignments in day_block_assignments.items():
                print(f"\n{day.name}, {block}:")
                for subject, room, time in assignments:
                    print(
                        f"  Subject: {subject.get_subject_name()} in {room} from {time._start_time} to {time._end_time}"
                    )
                    print(
                        f"    Instructors: {[instr.get_name() for instr in subject.get_subject_instructors()]}"
                    )
            print(total_conflicts)
            print(max_possible_conflicts)
            print(fitness)
        return fitness

    def generate_visual_schedule(self, block, filename="class_schedule.png"):
        schedule = {day: [] for day in Day}

        for b, subject, room, time_slot in self._assignments:
            if b == block:
                day = time_slot.get_day()
                start_time = time_slot._start_time
                end_time = time_slot._end_time
                schedule[day].append((start_time, end_time, subject, room))

        # Sort schedules for each day
        for day in Day:
            schedule[day].sort(key=lambda x: x[0])

        generate_visual_schedule(schedule, block, filename)


"""
Papers used: 
1. Preventing Premature Convergence in Genetic Algorithm
Using DGCA and Elitist Technique
2. An optimized solution to the course scheduling problem in universities under an improved genetic algorithm

Techniques used:
1. Elitism
2. Distributed Genetic Algorithm
3. Dynamic Genetic Clustering Algorithm
4. Multi-parent crossover
5. Multiple coevolution
6. Social Disaster Techniques (SDT)
"""


class ImprovedGeneticAlgorithm:
    def __init__(
        self,
        population_size: int,
        blocks: list[Block],
        rooms: list[Room],
        fitness_limit=1.00,
        num_subpopulations=5,
        migration_interval=10,
        migration_rate=0.1,
        elitism_rate=0.1,
    ):
        self._population_size = population_size
        self._blocks = blocks
        self._rooms = rooms
        self._fitness_limit = fitness_limit
        self._num_subpopulations = num_subpopulations
        self._migration_interval = migration_interval
        self._migration_rate = migration_rate
        self._elitism_rate = elitism_rate

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

    def _select_parent(self, subpopulation: list[Schedule]) -> Schedule:
        tournament_size = max(2, len(subpopulation) // 5)
        tournament = random.sample(subpopulation, tournament_size)
        return max(tournament, key=lambda schedule: schedule.calculate_fitness())

    def _crossover(
        self, parent1: Schedule, parent2: Schedule, strategy: str
    ) -> Schedule:
        child = Schedule(self._blocks, self._rooms)
        if strategy == "single_point":
            # Single-point crossover
            midpoint = len(parent1._assignments) // 2
            child._assignments = (
                parent1._assignments[:midpoint] + parent2._assignments[midpoint:]
            )
        elif strategy == "multi_parent":
            # Multi-parent crossover (example using 3 parents)
            parent3 = random.choice(
                self.subpopulations[0]
            )  # Example of selecting an additional parent
            third_point = len(parent1._assignments) // 3
            child._assignments = (
                parent1._assignments[:third_point]
                + parent2._assignments[third_point : 2 * third_point]
                + parent3._assignments[2 * third_point :]
            )
        return child

    def _mutate(self, schedule: Schedule, mutation_rate: float) -> None:
        for i in range(len(schedule._assignments)):
            if random.random() < mutation_rate:
                block, subject, _, time_slot = schedule._assignments[i]

                # Choose a random room
                room = random.choice(self._rooms)

                # Generate a new random start and end time using the helper function
                start_time, end_time = schedule.generate_random_time(subject)

                # Get the current days that the subject is already scheduled on
                current_days = [
                    assign[3].get_day()
                    for assign in schedule._assignments
                    if assign[0] == block and assign[1] == subject
                ]

                # Determine available days for mutation (days not yet assigned to this subject)
                available_days = list(set(Day) - set(current_days))

                # Choose a new day from the available days, or fallback to any day if no available days
                day = (
                    random.choice(available_days)
                    if available_days
                    else random.choice(list(Day))
                )

                # Create the new time slot with the mutated day
                new_time_slot = TimeSlot(day, start_time, end_time)

                # Update the current assignment with the new room and time slot
                schedule._assignments[i] = (block, subject, room, new_time_slot)

    def _migrate(self):
        for i in range(self._num_subpopulations):
            next_subpop = (i + 1) % self._num_subpopulations
            migrants = random.sample(
                self.subpopulations[i],
                int(len(self.subpopulations[i]) * self._migration_rate),
            )
            self.subpopulations[next_subpop].extend(migrants)
            self.subpopulations[next_subpop] = sorted(
                self.subpopulations[next_subpop],
                key=lambda x: x.calculate_fitness(),
                reverse=True,
            )[: len(self.subpopulations[i])]

    def _generate_child(self, parent1, parent2, crossover_strategy, subpop_idx):
        # Perform crossover to generate the child
        child = self._crossover(parent1, parent2, crossover_strategy)

        # Use subpopulation-specific mutation rates
        dynamic_mutation_rate = self.subpopulation_mutation_rates[subpop_idx]

        # Mutate the child
        self._mutate(child, dynamic_mutation_rate)

        return child

    def evolve(self, generations: int) -> tuple[list[tuple[int, Schedule]], float, int]:
        evolution_history = []
        start_time = t.time()

        for gen in range(generations):
            gen_start_time = t.time()
            for subpop_idx, subpopulation in enumerate(self.subpopulations):
                new_subpopulation = []

                # Elitism
                elites = int(len(subpopulation) * self._elitism_rate)
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
            if gen % self._migration_interval == 0:
                self._migrate()

            best_schedule = self.get_best_schedule()
            evolution_history.append((gen, best_schedule))

            # Stop if we've found a perfect solution
            if math.isclose(best_schedule.calculate_fitness(), self._fitness_limit):
                break

            # End timer for this generation and print results
            gen_end_time = t.time()
            gen_elapsed_time = gen_end_time - gen_start_time
            print(f"Generation {gen + 1} completed in {gen_elapsed_time:.2f} seconds.")

        elapsed_time = t.time() - start_time
        best_generation = evolution_history[-1][0]
        return evolution_history, elapsed_time, best_generation

    def get_best_schedule(self) -> Schedule:
        return max(
            (schedule for subpop in self.subpopulations for schedule in subpop),
            key=lambda schedule: schedule.calculate_fitness(),
        )

    def _calculate_diversity(self, subpopulation: list[Schedule]) -> float:
        fitnesses = [schedule.calculate_fitness() for schedule in subpopulation]
        avg_fitness = sum(fitnesses) / len(fitnesses)
        variance = sum((f - avg_fitness) ** 2 for f in fitnesses) / len(fitnesses)
        return (
            math.sqrt(variance) / avg_fitness
        )  # Coefficient of variation as a measure of diversity

    def _trigger_disaster_event(
        self, subpopulation: list[Schedule], disaster_rate: float = 0.5
    ):
        disaster_size = int(
            len(subpopulation) * disaster_rate
        )  # Replace a portion of the population
        new_individuals = [
            Schedule(self._blocks, self._rooms) for _ in range(disaster_size)
        ]
        subpopulation[-disaster_size:] = new_individuals

    def _detect_premature_convergence(self, subpopulation: list[Schedule]) -> bool:
        diversity = self._calculate_diversity(subpopulation)
        return diversity < 0.1  # Threshold for detecting premature convergence


def create_intructor(name: str) -> Instructor:
    return Instructor(name)


def create_subject(
    name: str, instructors: list[Instructor], subject_type: SubjectType
) -> Subject:
    return Subject(
        name=name, available_instructors=instructors, subject_type=subject_type
    )


def create_room(room_num: str, room_type: RoomType) -> Room:
    return Room(room_num=room_num, room_type=room_type)


def create_dept(
    prefix: str, year_level: YearLevel, subjects: list[Subject]
) -> Department:
    year_level_subjects = {year_level: subjects}
    return Department(
        prefix=prefix, year_level_subjects=year_level_subjects, year_level=year_level
    )


def create_block(
    block_num: str, department: Department, year_level: YearLevel
) -> Block:
    return Block(block_num, department, year_level)


if __name__ == "__main__":
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

    IMODSIM = Subject("IMODSIM", [maam_lou, sir_marc], SubjectType.SPECIALIZED_LECTURE)
    PROBSTAT = Subject("PROBSTAT", [sir_lloyd, mam_max], SubjectType.GENERAL_EDUCATION)
    INTCALC = Subject("INTCALC", [sir_glenn, anisa], SubjectType.GENERAL_EDUCATION)
    ATF = Subject("ATF", [sir_uly, patrick], SubjectType.SPECIALIZED_LECTURE)
    SOFTENG = Subject("SOFTENG", [maam_raquel, kyle], SubjectType.SPECIALIZED_LECTURE)
    ADET = Subject("ADET", [kyle, sir_uly], SubjectType.SPECIALIZED_LAB)

    subjects = [IMODSIM, PROBSTAT, INTCALC, ATF, SOFTENG, PROBSTAT]

    ROOMS = [
        ["SJH-503", RoomType.LECTURE],
        ["SJH-504", RoomType.LAB],
        ["SJH-505", RoomType.LECTURE],
        ["SJH-506", RoomType.LAB],
    ]

    dept = Department("CS", {YearLevel.THIRD: subjects}, YearLevel.THIRD)

    rooms = [Room(room_num, room_type) for room_num, room_type in ROOMS]

    block1 = Block("301", dept, YearLevel.THIRD)
    block2 = Block("302", dept, YearLevel.THIRD)
    block3 = Block("303", dept, YearLevel.THIRD)
    block4 = Block("304", dept, YearLevel.THIRD)

    ga = ImprovedGeneticAlgorithm(
        population_size=350,
        blocks=[block1, block2, block3, block4],
        rooms=rooms,
        num_subpopulations=10,
        migration_interval=10,
        migration_rate=0.1,
        elitism_rate=0.1,
    )

    evolution_history, elapsed_time, best_generation = ga.evolve(generations=350)

    print("\nEvolution completed.")
    best_schedule = evolution_history[-1][1]  # Get the best schedule
    for idx, block in enumerate([block1, block2, block3, block4]):
        # best_schedule.print_block_schedule(block)
        best_schedule.generate_visual_schedule(block, f"block{idx+1}_schedule.png")

    best_schedule = ga.get_best_schedule()
    print(f"\nFitness: {best_schedule.calculate_fitness():.4f}")
    print(f"Best solution found in generation: {best_generation}")
    print(f"Time taken to find the best solution: {elapsed_time:.2f} seconds")
