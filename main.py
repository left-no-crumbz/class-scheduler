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
#  - A subject should be able to occupy decimal hours (e.g., 1 hour and 30 minutes, etc)
#  - A subject can have multiple available instructors and it randomly selects from the list of instructors.
#  - If a subject has both Lecture and Lab components, it should use a Lab room in the first
#    meeting in a week, and Lecture room in the second week or vice versa.


import time as t
from datetime import datetime, time, timedelta
from enum import Enum, IntEnum

import numpy as np

# TODO: Add a subject type that is Specialized, General Ed, etc.
# pure lecture: 1 hr 30 mins
# lab: 3 hours
# lecture ng lab: 2 hours
# I guess we can do:
# If subjectType == BOTH then it could appear only once


class RoomType(Enum):
    LECTURE = 1
    LAB = 2


class SubjectType(Enum):
    LECTURE = 1
    LAB = 2
    BOTH = 3


class YearLevel(IntEnum):
    FIRST = 1
    SECOND = 2
    THIRD = 3
    FOURTH = 4


class Day(IntEnum):
    MONDAY = 1
    TUESDAY = 2
    WEDNESDAY = 3
    THURSDAY = 4
    FRIDAY = 5


class Instructors:
    def __init__(self, name: str) -> None:
        self.name = name

    def get_name(self) -> str:
        return self.name

    def __str__(self) -> str:
        return f"Instructor: {self.name}"


class Room:
    def __init__(self, room_num: str, max_capacity: int, room_type: RoomType) -> None:
        self.room_num = room_num
        self.max_capacity = max_capacity
        self.room_type = room_type

    def get_room_num(self) -> str:
        return self.room_num

    def get_max_capacity(self) -> int:
        return self.max_capacity

    def get_room_type(self) -> RoomType:
        return self.room_type

    def get_room_type_str(self) -> str:
        return self.room_type.name

    def __str__(self) -> str:
        return f"Room Number: {self.room_num}, Max Capacity: {self.max_capacity}, Room Type: {self.room_type.name}"


class Subject:
    def __init__(
        self,
        name: str,
        instructors: list[Instructors],
        subject_type: SubjectType,
        duration: timedelta,
    ) -> None:
        self.name = name
        self.instructors = instructors
        self.subject_type = subject_type
        self.duration = duration

    def get_subject_name(self) -> str:
        return self.name

    def get_subject_type(self) -> SubjectType:
        return self.subject_type

    def get_instructors(self) -> list[Instructors]:
        return self.instructors

    def get_subject_duration(self) -> timedelta:
        return self.duration

    def __str__(self) -> str:
        instructors_str = ", ".join(
            instructor.get_name() for instructor in self.instructors
        )
        return f"Subject: {self.name}, Type: {self.subject_type.name}, Instructors: [{instructors_str}], Duration: {self.duration}"


class Department:
    def __init__(self, prefix: str, subjects: dict[YearLevel, list[Subject]]) -> None:
        self.prefix = prefix  # CS, WD, NA, EMC, CYB

        # Offered subjects by each department depending on their yearlevel
        # i.e., DASALGO, ADMATHS for CS
        # ANAGEOM for WD, etc
        self.subjects = subjects

    def get_dept_prefix(self) -> str:
        return self.prefix

    def get_dept_subjects(self) -> dict[YearLevel, list[Subject]]:
        return self.subjects

    def __str__(self) -> str:
        str_dict = {
            k: [subject.get_subject_name() for subject in v]
            for k, v in self.subjects.items()
        }
        return f"Prefix: {self.prefix}, Offered Subjects Per Year Level: [{str_dict}]"


class Block:
    def __init__(
        self,
        year_level: YearLevel,
        dept: Department,
        num_student: int,
    ) -> None:
        self.year_level = year_level
        self.dept = dept
        self.subjects = self.get_subjects(year_level)
        self.num_students = num_student
        self.block_num = self.generate_block_num(year_level.value)

    counters = {}

    @classmethod
    def generate_block_num(cls, year_level: int) -> int:
        if year_level not in cls.counters:
            cls.counters[year_level] = 1
        else:
            cls.counters[year_level] += 1
        return int(f"{year_level}{cls.counters[year_level]:02d}")

    def get_year_level(self) -> YearLevel:
        return self.year_level

    def get_dept(self) -> Department:
        return self.dept

    def get_subjects(self, year_level: YearLevel) -> list[Subject]:
        return self.dept.get_dept_subjects()[year_level]

    def get_num_students(self) -> int:
        return self.num_students

    def get_block_num(self) -> int:
        return self.block_num


class TimeSlot:
    def __init__(
        self, day: Day, start_time: timedelta, end_time: timedelta, buffer: timedelta
    ) -> None:
        self.day = day
        self.start_time = start_time
        self.end_time = end_time
        self.buffer = buffer  # The buffer after each subject

    def get_day(self) -> Day:
        return self.day

    def get_start_time(self) -> timedelta:
        return self.start_time

    def get_end_time(self) -> timedelta:
        return self.end_time

    def get_buffer(self) -> timedelta:
        return self.buffer


# Schedule is the representation of the genome
class Schedule:
    def __init__(self, blocks: list[Block], rooms: list[Room]) -> None:
        self.blocks = np.array(blocks, dtype=object)
        self.rooms = np.array(rooms, dtype=object)
        self.assignments = []
        self.generate_random_schedule()
        self.calculate_fitness()

    # the function to generate new solutions
    def generate_random_schedule(self):
        # Create a Random Number Generator
        rng = np.random.default_rng()

        #  Calculate the total number of subjects
        block_subjects = np.array([len(block.subjects) for block in self.blocks])
        total_subjects = np.sum(block_subjects)

        # Calculate random rooms and days in bulk
        random_rooms = rng.choice(self.rooms, total_subjects)
        random_days = rng.choice(list(Day), total_subjects)

        EARLIEST_START = time(7, 0)
        LATEST_END = time(21, 0)

        for block in self.blocks:
            for i, subject in enumerate(block.subjects):
                # create a room from an array of random rooms
                room = random_rooms[i]

                # choose a day from an array of random days
                day = random_days[i]

                # generate a random minute, highest is 14 hours
                # 14 hours is the span from EARLIEST_START (7AM) to LATEST_END (9PM)
                random_minutes = rng.integers(0, 14 * 60)

                # generate a start time
                start_time = (
                    datetime.combine(datetime.min, EARLIEST_START)
                    + timedelta(minutes=float(random_minutes))
                ).time()

                # generate an end time
                end_time = (
                    datetime.combine(datetime.min, start_time) + subject.duration
                ).time()

                # adjust the end time if its past 9pm
                if end_time > LATEST_END:
                    end_time = LATEST_END
                    start_time = (
                        datetime.combine(datetime.min, LATEST_END) - subject.duration
                    ).time()

                # Convert the times into timedelta
                start_timedelta = timedelta(
                    hours=start_time.hour, minutes=start_time.minute
                )
                end_timedelta = timedelta(hours=end_time.hour, minutes=end_time.minute)

                time_slot = TimeSlot(
                    day, start_timedelta, end_timedelta, buffer=timedelta(0)
                )
                self.assignments.append((block, subject, room, time_slot))

    def calculate_fitness(self):
        conflicts = 0
        total_assignments = len(self.assignments)
        max_allowed_conflicts = total_assignments * 0.6

        # unpack the values
        blocks = np.array([assignment[0] for assignment in self.assignments])
        subjects = np.array([assignment[1] for assignment in self.assignments])
        rooms = np.array([assignment[2] for assignment in self.assignments])
        time_slots = np.array([assignment[3] for assignment in self.assignments])

        # precompute values
        instructor_sets = np.array(
            [set(subject.get_instructors()) for subject in subjects]
        )
        days = np.array([ts.get_day() for ts in time_slots])

        # Helper function to check if two time ranges overlap
        def time_overlap(time1: TimeSlot, time2: TimeSlot) -> bool:
            return (
                time1.start_time < time2.end_time and time2.start_time < time1.end_time
            )

        for i in range(total_assignments):
            for j in range(i + 1, total_assignments):
                # Check for day and time conflicts
                if days[i] == days[j] and time_overlap(time_slots[i], time_slots[j]):
                    conflicts += 1

                # Check for room conflicts
                if rooms[i] == rooms[j]:
                    conflicts += 1

                # Check for instructor conflicts
                if not instructor_sets[i].isdisjoint(instructor_sets[j]):
                    conflicts += 1

                # Check for block conflicts
                if blocks[i] == blocks[j]:
                    conflicts += 1
            if conflicts > max_allowed_conflicts:
                return 0
        normalized_conflicts = conflicts / total_assignments
        # print(1 / (1 + normalized_conflicts))
        return 1 / (1 + normalized_conflicts)  # fitness


def create_schedule(blocks, rooms):
    return Schedule(blocks, rooms)


class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int,
        mutation_rate: float,
        blocks: list[Block],
        rooms: list[Room],
        fitness_limit=1.00,
    ) -> None:
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.blocks = blocks
        self.rooms = rooms
        self.population = self.generate_population()
        self.vblock = np.array(self.blocks)
        self.vroom = np.array(self.rooms)
        self.vpopulation = np.array(self.population)
        self.fitness_limit = fitness_limit
        self.vectorized_fitness_func = np.frompyfunc(
            lambda s: s.calculate_fitness(), 1, 1
        )
        self.parent = self.select_parents()

    def generate_population(self) -> list[Schedule]:
        start_time = t.time()  # Start timing

        population = [
            Schedule(self.blocks, self.rooms) for _ in range(self.population_size)
        ]

        end_time = t.time()  # End timing
        print(f"Time taken to generate population: {end_time - start_time} seconds")
        return population

    def select_parents(self):
        fitness_list = np.array(
            self.vectorized_fitness_func(self.population), dtype=np.float64
        )
        top_parent = np.argmax(fitness_list)
        return top_parent


if __name__ == "__main__":
    maam_mags = Instructors("Ma'am Avigail Magbag")
    maam_jehan = Instructors("Ma'am Jehan")
    maam_caro = Instructors("Ma'am Caro")
    sir_uly = Instructors("Sir Uly")
    maam_lou = Instructors("Ma'am Lou")

    # TODO: Attach the duration to the SubjectType
    LOGPROG_LECTURE = Subject(
        "LOGPROG", [maam_mags], SubjectType.LECTURE, timedelta(hours=3)
    )
    LOGPROG_LAB = Subject(
        "LOGPROG", [maam_mags], SubjectType.LAB, timedelta(hours=1, minutes=30)
    )
    PHCI = Subject(
        "PHCI", [maam_jehan], SubjectType.LECTURE, timedelta(hours=1, minutes=30)
    )
    OOP_LECTURE = Subject(
        "OOP", [maam_caro], SubjectType.BOTH, timedelta(hours=1, minutes=30)
    )
    OOP_LAB = Subject("OOP", [maam_caro], SubjectType.BOTH, timedelta(hours=3))

    ATF = Subject("ADET", [sir_uly], SubjectType.LAB, timedelta(hours=3))

    ATF = Subject("ATF", [sir_uly], SubjectType.LECTURE, timedelta(hours=2))

    IMODSIM = Subject("IMODSIM", [maam_lou], SubjectType.LECTURE, timedelta(hours=2))

    dept = Department(
        "CS",
        {
            YearLevel.FIRST: [LOGPROG_LECTURE, LOGPROG_LAB, PHCI],
            YearLevel.SECOND: [OOP_LAB, OOP_LECTURE],
            YearLevel.THIRD: [ATF, IMODSIM],
        },
    )

    ROOMS = [
        ["SJH-503", 45, RoomType.LECTURE],
        ["SJH-504", 45, RoomType.LAB],
        ["SJH-505", 45, RoomType.LECTURE],
    ]

    block1 = Block(YearLevel.THIRD, dept, 45)
    block2 = Block(YearLevel.THIRD, dept, 45)
    block3 = Block(YearLevel.THIRD, dept, 45)

    rooms = [
        Room(room_num, capacity, room_type) for room_num, capacity, room_type in ROOMS
    ]
    sched = Schedule([block1, block2, block3], rooms)

    ga = GeneticAlgorithm(
        population_size=100,
        mutation_rate=0.2,
        blocks=[block1, block2, block3],
        rooms=rooms,
    )

    print(Schedule)
