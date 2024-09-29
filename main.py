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

import math
import random
import time
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from typing import List

import numpy as np
from tabulate import tabulate

#  TODO:
# [ ] - implement room types
# [ ] - implement additional time constraints
# [ ] - print the table wherein each class would have their own printed schedule


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

        # Offered subjects by each department
        # i.e., DASALGO, ADMATHS for CS
        # ANAGEOM for WD, etc
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

        # This is slightly different from what subjects a department offers.
        # This is because a class can have a variety of subjects (i.e, Gen Ed, etc)
        # This refers to the list of subjects that a class is taking.
        self._subjects = subjects

        # The number students in a block.
        # A room can only accomodate one block per time slot.
        # We are putting an assumption that the # of students in a block
        # will always fill and fit in a room.
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


# TODO: This should support minutes as well instead of full hours
# TODO: This should be in 12-hour format
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
                room = random.choice(self.rooms)

                # Generate a random start time between 7:00 AM and 9:00 PM
                start_time = datetime.combine(
                    datetime.today(), datetime.min.time()
                ) + timedelta(hours=7, minutes=random.randint(0, 14 * 60))

                end_time = start_time + subject.get_subject_duration()

                # If end time is after 9:00 PM, adjust start time
                if end_time.hour >= 21:
                    start_time = (
                        datetime.combine(datetime.today(), datetime.min.time())
                        + timedelta(hours=21)
                        - subject.get_subject_duration()
                    )
                    end_time = start_time + subject.get_subject_duration()

                day = random.choice(list(Day))
                time_slot = TimeSlot(day, start_time, end_time)
                self.assignments.append((block, subject, room, time_slot))

    def calculate_fitness(self) -> float:
        conflicts = 0
        total_assignments = len(self.assignments)

        # Helper function to check if two time ranges overlap
        def time_overlap(time1: TimeSlot, time2: TimeSlot) -> bool:
            return (
                time1.start_time < time2.end_time and time2.start_time < time1.end_time
            )

        # Helper function to calculate time difference in minutes
        def time_diff_minutes(time1: datetime, time2: datetime) -> int:
            return abs(int((time1 - time2).total_seconds() / 60))

        for i, (block1, subject1, room1, time1) in enumerate(self.assignments):
            day1 = time1.get_day()

            # Check if subject is within 7am to 9pm
            if time1.start_time.hour < 7 or time1.end_time.hour >= 21:
                conflicts += 1

            # Check for conflicts with other assignments
            for j, (block2, subject2, room2, time2) in enumerate(
                self.assignments[i + 1 :], i + 1
            ):
                day2 = time2.get_day()

                if day1 == day2 and time_overlap(time1, time2):
                    # Room conflict
                    if room1 == room2:
                        conflicts += 1

                    # Instructor conflict
                    if any(
                        instr in subject2.get_subject_instructors()
                        for instr in subject1.get_subject_instructors()
                    ):
                        conflicts += 1

                    # Block conflict
                    if block1 == block2:
                        conflicts += 1

            # Check for proper intervals between subjects
            block_assignments = [
                a for a in self.assignments if a[0] == block1 and a[3].get_day() == day1
            ]
            sorted_assignments = sorted(
                block_assignments, key=lambda a: a[3].start_time
            )

            for k in range(len(sorted_assignments) - 1):
                _, _, _, current_slot = sorted_assignments[k]
                _, _, _, next_slot = sorted_assignments[k + 1]

                interval = time_diff_minutes(
                    next_slot.start_time, current_slot.end_time
                )

                # Penalize if interval is less than 5 minutes or more than 10 minutes
                if interval < 5 or interval > 10:
                    conflicts += 1

        # Check for breaks after every 5 hours
        for block in self.blocks:
            for day in Day:
                day_assignments = sorted(
                    [
                        a
                        for a in self.assignments
                        if a[0] == block and a[3].get_day() == day
                    ],
                    key=lambda a: a[3].start_time,
                )

                if day_assignments:
                    cumulative_time = timedelta()
                    last_break_end = day_assignments[0][3].start_time

                    for _, _, _, time_slot in day_assignments:
                        cumulative_time += time_slot.end_time - time_slot.start_time

                        if cumulative_time >= timedelta(hours=5):
                            break_duration = time_diff_minutes(
                                time_slot.end_time, last_break_end
                            )
                            if break_duration < 30 or break_duration > 60:
                                conflicts += 1
                            cumulative_time = timedelta()
                            last_break_end = time_slot.end_time

        # Normalize conflicts
        normalized_conflicts = conflicts / total_assignments

        # Calculate fitness (higher is better)
        fitness = 1 / (1 + normalized_conflicts)

        return fitness

    def print_block_schedule(self, block: Block):
        schedule = {day: {} for day in Day}

        for b, subject, room, time_slot in self.assignments:
            if b == block:
                day = time_slot.get_day()
                start_time = time_slot.start_time
                end_time = time_slot.end_time
                schedule[day][(start_time, end_time)] = (
                    f"{subject.get_subject_name()}\n{room.get_room_num()}"
                )

        # Generate all possible time slots
        all_time_slots = []
        current_time = datetime.combine(
            datetime.today(), datetime.min.time()
        ) + timedelta(hours=7)
        end_of_day = datetime.combine(
            datetime.today(), datetime.min.time()
        ) + timedelta(hours=21)

        while current_time < end_of_day:
            next_time = current_time + timedelta(minutes=30)
            all_time_slots.append((current_time, next_time))
            current_time = next_time

        table_data = []
        for start_time, end_time in all_time_slots:
            row = [
                f"{start_time.strftime('%I:%M %p')} - {end_time.strftime('%I:%M %p')}"
            ]
            for day in Day:
                cell_content = ""
                for (class_start, class_end), content in schedule[day].items():
                    if class_start <= start_time < class_end:
                        cell_content = content
                        break
                row.append(cell_content)
            table_data.append(row)

        # Print the table
        headers = ["Time"] + [day.name.capitalize() for day in Day]
        print(
            f"\nSchedule for {block.get_block_dept().get_dept_prefix()}-{block.get_block_number()}:"
        )
        print(tabulate(table_data, headers=headers, tablefmt="grid"))


class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int,
        mutation_rate: float,
        blocks: List[Block],
        rooms: List[Room],
        fitness_limit=1.00,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.blocks = blocks
        self.rooms = rooms
        self.population = [Schedule(blocks, rooms) for _ in range(population_size)]
        self.fitness_limit = fitness_limit
        self.vcalc_fit = np.frompyfunc(lambda s: s.calculate_fitness(), 1, 1)

    def _select_parent(self) -> Schedule:
        # Arbitrary precision level
        precision_level = 0.5

        # Yamane's Formula
        sample_size = self.population_size / (
            1 + (self.population_size * (precision_level**2))
        )

        # select a sample from the given population and calculated sample size
        tournament = random.sample(self.population, math.floor(sample_size))

        # Calculate fitness for all schedules in the tournament
        fitness_values = self.vcalc_fit(tournament)

        # Find the index of the maximum fitness value
        max_fitness_index = np.argmax(fitness_values)

        # Return the schedule with the highest fitness
        return tournament[max_fitness_index]

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

                # Generate a random start time between 7:00 AM and 9:00 PM
                start_time = datetime.combine(
                    datetime.today(), datetime.min.time()
                ) + timedelta(hours=7, minutes=random.randint(0, 14 * 60))

                end_time = start_time + subject.get_subject_duration()

                # If end time is after 9:00 PM, adjust start time
                if end_time.hour >= 21:
                    start_time = (
                        datetime.combine(datetime.today(), datetime.min.time())
                        + timedelta(hours=21)
                        - subject.get_subject_duration()
                    )
                    end_time = start_time + subject.get_subject_duration()

                day = random.choice(list(Day))
                time_slot = TimeSlot(day, start_time, end_time)
                schedule.assignments[i] = (block, subject, room, time_slot)

    def evolve(self, generations: int):
        evolution_history = []
        start_time = time.time()
        for gen in range(generations):
            new_population = []
            for _ in range(self.population_size):
                parent1 = self._select_parent()
                parent2 = self._select_parent()
                child = self._crossover(parent1, parent2)
                self._mutate(child)
                new_population.append(child)
            self.population = new_population

            best_schedule = self.get_best_schedule()
            evolution_history.append((gen, best_schedule))

            # Stop if we've found a perfect solution
            if math.isclose(best_schedule.calculate_fitness(), self.fitness_limit):
                break

        elapsed_time = time.time() - start_time
        best_generation = evolution_history[-1][0]
        return evolution_history, elapsed_time, best_generation

    def get_best_schedule(self) -> Schedule:
        return max(self.population, key=lambda schedule: schedule.calculate_fitness())


if __name__ == "__main__":
    # Create sample data (instructors, subjects, departments, blocks, rooms)
    sir_uly = Instructor("Sir Ulysses Monsale")
    maam_lou = Instructor("Ma'am Louella Salenga")
    sir_glenn = Instructor("Sir Glenn Mañalac")
    sir_lloyd = Instructor("Sir Lloyd Estrada")
    maam_raquel = Instructor("Ma'am Raquel Rivera")

    IMODSIM = Subject("IMODSIM", [maam_lou], timedelta(hours=2))
    PROBSTAT = Subject("PROBSTAT", [sir_lloyd], timedelta(hours=1, minutes=30))
    INTCALC = Subject("INTCALC", [sir_glenn], timedelta(hours=1, minutes=30))
    ATF = Subject("ATF", [sir_uly], timedelta(hours=3))
    SOFTENG = Subject("SOFTENG", [maam_raquel], timedelta(hours=2))

    ROOMS = [
        ["SJH-503", 45, RoomType.LECTURE],
        ["SJH-504", 45, RoomType.LAB],
        ["SJH-505", 45, RoomType.LECTURE],
    ]

    dept = Department("CS", [INTCALC, PROBSTAT, IMODSIM, ATF, SOFTENG])
    dept_subjects = dept.get_dept_subjects()

    block1 = Block("301", dept, dept_subjects, 45)
    block2 = Block("302", dept, dept_subjects, 45)
    block3 = Block("303", dept, dept_subjects, 45)

    rooms = [
        Room(room_num, capacity, room_type) for room_num, capacity, room_type in ROOMS
    ]

    ga = GeneticAlgorithm(
        population_size=1000,
        mutation_rate=0.2,
        blocks=[block1, block2, block3],
        rooms=rooms,
    )

    print("Initial population created.")
    print(f"Number of schedules in population: {len(ga.population)}")
    print(
        f"Number of assignments in first schedule: {len(ga.population[0].assignments)}"
    )

    evolution_history, elapsed_time, best_generation = ga.evolve(generations=100)

    print("\nEvolution completed.")
    best_schedule = evolution_history[-1][1]  # Get the best schedule
    for block in [block1, block2, block3]:
        best_schedule.print_block_schedule(block)

    print(f"\nFitness: {best_schedule.calculate_fitness():.4f}")
    print(f"Best solution found in generation: {best_generation}")
    print(f"Time taken to find the best solution: {elapsed_time:.2f} seconds")
