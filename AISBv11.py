import pandas as pd
from ortools.sat.python import cp_model
import os
import re
import sys
import time

# --- GLOBAL SETTINGS (can be adjusted) ---
MAX_SECTIONS_PER_COURSE = 8 

# --- Data Loading Functions ---
def load_data(student_requests_filepath, disallowed_periods_filepath, course_definitions_filepath, disallowed_course_pairs_filepath=None, teacher_data_filepath=None):
    """
    Loads all necessary data from CSV files.
    """
    student_requests, course_definitions, disallowed_course_period_combinations, disallowed_course_pairs, teachers = [], {}, [], [], {}

    # --- Load Student Requests ---
    try:
        df_requests = pd.read_csv(student_requests_filepath)
        for _, row in df_requests.iterrows():
            requests = [str(row[col]).strip() for col in df_requests.columns if col.startswith('course_') and pd.notna(row[col])]
            student_requests.append({"student_id": str(row["student_id"]).strip(), "name": str(row["name"]).strip(), "requests": requests})
        print(f"Loaded {len(student_requests)} student requests from {student_requests_filepath}", flush=True)
    except Exception as e:
        sys.exit(f"ERROR loading student requests: {e}")

    all_periods = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]

    # --- Load Course Definitions ---
    try:
        df_course_defs = pd.read_csv(course_definitions_filepath)
        for _, row in df_course_defs.iterrows():
            course_name = str(row["course_name"]).strip()
            course_definitions[course_name] = {
                "total_sections": int(row["total_sections"]),
                "max_capacity": int(row["max_capacity_per_section"]),
                "allow_multiple_sections_in_same_period": str(row.get("allow_multiple_sections_in_same_period", "n")).strip().lower() == 'y'
            }
        print(f"Loaded {len(course_definitions)} course definitions.", flush=True)
    except Exception as e:
        sys.exit(f"ERROR loading course definitions: {e}")

    # --- Load Disallowed Course-Period Combinations ---
    try:
        df_disallowed = pd.read_csv(disallowed_periods_filepath)
        for _, row in df_disallowed.iterrows():
            if pd.notna(row.get("course_name")) and pd.notna(row.get("period")):
                disallowed_course_period_combinations.append((str(row["course_name"]).strip(), str(row["period"]).strip()))
        print(f"Loaded {len(disallowed_course_period_combinations)} disallowed course-period combinations.", flush=True)
    except FileNotFoundError:
        print("WARNING: disallowed_periods.csv not found.", flush=True)
    except Exception as e:
        sys.exit(f"ERROR loading disallowed periods: {e}")
        
    # --- Load Disallowed Course Pairs ---
    if disallowed_course_pairs_filepath:
        try:
            df_disallowed_pairs = pd.read_csv(disallowed_course_pairs_filepath)
            for _, row in df_disallowed_pairs.iterrows():
                course1 = str(row.get("course_1")).strip()
                course2 = str(row.get("course_2")).strip()
                if pd.notna(course1) and pd.notna(course2):
                    if course1 in course_definitions and course2 in course_definitions:
                        disallowed_course_pairs.append((course1, course2))
            print(f"Loaded {len(disallowed_course_pairs)} disallowed course pairs.", flush=True)
        except FileNotFoundError:
            print(f"WARNING: disallowed_course_pairs.csv not found.", flush=True)
        except Exception as e:
            sys.exit(f"ERROR loading disallowed course pairs: {e}")

    # --- Load Teacher Data (for Stage 2)---
    if teacher_data_filepath:
        try:
            df_teachers = pd.read_csv(teacher_data_filepath)
            for _, row in df_teachers.iterrows():
                teacher_name = str(row["teacher_name"]).strip()
                subjects = [s.strip() for s in str(row["subjects_taught"]).split('|') if s.strip()]
                teachers[teacher_name] = { "subjects_taught": [s for s in subjects if s in course_definitions] }
            print(f"Loaded {len(teachers)} teachers for Stage 2 assignment.", flush=True)
        except FileNotFoundError:
            print("WARNING: teacher_data.csv not found. Teachers will not be assigned.", flush=True)
        except Exception as e:
            sys.exit(f"ERROR loading teacher data: {e}")

    return student_requests, course_definitions, all_periods, disallowed_course_period_combinations, disallowed_course_pairs, teachers

# --- Helper to extract section index from "Course (Sec X)" or "SecN" ---
def _get_section_index_from_id(section_id_str):
    """Extracts 0-indexed section number from a string like 'Course Name (Sec N)' or 'SecN'."""
    # Try matching the "Course (Sec N)" format
    match = re.search(r'\(Sec (\d+)\)', section_id_str)
    if match:
        return int(match.group(1)) - 1 # Convert "Sec 1" to 0-indexed 0

    # If not found, try matching the "SecN" format
    match = re.search(r'Sec(\d+)', section_id_str)
    if match:
        return int(match.group(1)) - 1 # Convert "Sec 1" to 0-indexed 0

    return None

# --- STAGE 1: Build Master Schedule (Students Only) ---
def build_master_schedule_model(student_requests, course_definitions, all_periods, disallowed_course_period_combinations, disallowed_course_pairs):
    model = cp_model.CpModel()

    # Variables
    is_section_active = {
        c_name: {
            s_idx: {p: model.NewBoolVar(f'active_{c_name}_s{s_idx}_p{p}') for p in all_periods}
            for s_idx in range(MAX_SECTIONS_PER_COURSE)
        } for c_name in course_definitions
    }

    student_assigned_to_section = {
        s["student_id"]: {
            c_name: {
                s_idx: model.NewBoolVar(f'assign_{s["student_id"]}_{c_name}_s{s_idx}')
                for s_idx in range(MAX_SECTIONS_PER_COURSE)
            } for c_name in s["requests"] if c_name in course_definitions
        } for s in student_requests
    }
    
    student_in_course_period = {
        s["student_id"]: {
            c_name: {
                p: model.NewBoolVar(f'stud_{s["student_id"]}_{c_name}_p{p}')
                for p in all_periods
            } for c_name in s["requests"] if c_name in course_definitions
        } for s in student_requests
    }

    # --- Constraints ---
    # A student can take at most one course in any given period
    for student in student_requests:
        s_id = student["student_id"]
        for p in all_periods:
            model.AddAtMostOne(student_in_course_period[s_id][c_name][p] for c_name in student_in_course_period[s_id])

    # The sum of active sections across all periods for a course must equal its total_sections
    for c_name, defs in course_definitions.items():
        model.Add(sum(is_section_active[c_name][s][p] for s in range(MAX_SECTIONS_PER_COURSE) for p in all_periods) == defs["total_sections"])

    # A specific section can only be active in one period
    for c_name in course_definitions:
        for s_idx in range(MAX_SECTIONS_PER_COURSE):
            model.AddAtMostOne(is_section_active[c_name][s_idx][p] for p in all_periods)

    # Enrollment in any section cannot exceed its max_capacity
    for c_name, defs in course_definitions.items():
        for s_idx in range(MAX_SECTIONS_PER_COURSE):
            students_in_section = [student_assigned_to_section[s["student_id"]][c_name][s_idx] for s in student_requests if c_name in student_assigned_to_section.get(s["student_id"], {})]
            model.Add(sum(students_in_section) <= defs["max_capacity"])

    # A course cannot be scheduled in a disallowed period
    for dis_course, dis_period in disallowed_course_period_combinations:
        if dis_course in course_definitions:
            for s_idx in range(MAX_SECTIONS_PER_COURSE):
                model.Add(is_section_active[dis_course][s_idx][dis_period] == 0)

    # A student can be assigned to at most one section of a requested course
    for student in student_requests:
        s_id = student["student_id"]
        for c_name in student["requests"]:
            if c_name in student_assigned_to_section.get(s_id, {}):
                model.AddAtMostOne(student_assigned_to_section[s_id][c_name][s_idx] for s_idx in range(MAX_SECTIONS_PER_COURSE))
    
    for student in student_requests:
        s_id = student["student_id"]
        for c_name in student["requests"]:
            if c_name not in student_in_course_period.get(s_id,{}): continue
            for p in all_periods:
                assignments_in_period = []
                for s_idx in range(MAX_SECTIONS_PER_COURSE):
                    link = model.NewBoolVar(f'link_{s_id}_{c_name}_s{s_idx}_p{p}')
                    model.AddBoolAnd([student_assigned_to_section[s_id][c_name][s_idx], is_section_active[c_name][s_idx][p]]).OnlyEnforceIf(link)
                    assignments_in_period.append(link)
                model.Add(sum(assignments_in_period) >= 1).OnlyEnforceIf(student_in_course_period[s_id][c_name][p])
                model.Add(sum(assignments_in_period) == 0).OnlyEnforceIf(student_in_course_period[s_id][c_name][p].Not())

    for c_name, defs in course_definitions.items():
        if not defs["allow_multiple_sections_in_same_period"]:
            for p in all_periods:
                model.Add(sum(is_section_active[c_name][s_idx][p] for s_idx in range(MAX_SECTIONS_PER_COURSE)) <= 1)

    for c1, c2 in disallowed_course_pairs:
        if c1 in course_definitions and c2 in course_definitions:
            for p in all_periods:
                c1_active_in_p = model.NewBoolVar(f'is_active_{c1}_{p}')
                sum_c1_sections = sum(is_section_active[c1][s_idx][p] for s_idx in range(MAX_SECTIONS_PER_COURSE))
                model.Add(sum_c1_sections > 0).OnlyEnforceIf(c1_active_in_p)
                model.Add(sum_c1_sections == 0).OnlyEnforceIf(c1_active_in_p.Not())

                c2_active_in_p = model.NewBoolVar(f'is_active_{c2}_{p}')
                sum_c2_sections = sum(is_section_active[c2][s_idx][p] for s_idx in range(MAX_SECTIONS_PER_COURSE))
                model.Add(sum_c2_sections > 0).OnlyEnforceIf(c2_active_in_p)
                model.Add(sum_c2_sections == 0).OnlyEnforceIf(c2_active_in_p.Not())
                model.AddAtMostOne([c1_active_in_p, c2_active_in_p])

    # Objective: Maximize the number of fulfilled student requests
    fulfilled_vars = [var for s in student_in_course_period.values() for c in s.values() for var in c.values()]
    model.Maximize(sum(fulfilled_vars))

    return model, {"is_section_active": is_section_active, "student_assigned_to_section": student_assigned_to_section, "student_in_course_period": student_in_course_period}

# --- Function for Rebuilding and Resolving Model with Fixed Sections ---
def build_model_for_reschedule(current_fixed_offerings, student_requests, course_definitions, all_periods, disallowed_course_period_combinations, disallowed_course_pairs, section_to_swap_info=None):
    """
    Builds a CP-SAT model for rescheduling, fixing all sections except the one being swapped,
    and then optimizing student assignments.

    Args:
        current_fixed_offerings (dict): The dictionary representing the current schedule,
                                         which includes all sections and their periods.
        student_requests (list): List of student request dictionaries.
        course_definitions (dict): Dictionary of course definitions.
        all_periods (list): List of all periods.
        disallowed_course_period_combinations (list): List of (course, period) tuples that are disallowed.
        disallowed_course_pairs (list): List of (course1, course2) tuples that cannot be in the same period.
        section_to_swap_info (dict, optional): Information about the section being swapped.
                                                Expected keys: 'course_name', 's_idx', 'old_period', 'new_period'.
                                                If None, all sections from current_fixed_offerings are fixed.
    
    Returns:
        tuple: (cp_model.CpModel, dict) - The constructed model and its variables.
    """
    model = cp_model.CpModel()

    # Re-declare variables
    is_section_active = {
        c_name: {
            s_idx: {p: model.NewBoolVar(f'active_{c_name}_s{s_idx}_p{p}') for p in all_periods}
            for s_idx in range(MAX_SECTIONS_PER_COURSE)
        } for c_name in course_definitions
    }

    student_assigned_to_section = {
        s["student_id"]: {
            c_name: {
                s_idx: model.NewBoolVar(f'assign_{s["student_id"]}_{c_name}_s{s_idx}')
                for s_idx in range(MAX_SECTIONS_PER_COURSE)
            } for c_name in s["requests"] if c_name in course_definitions
        } for s in student_requests
    }
    
    student_in_course_period = {
        s["student_id"]: {
            c_name: {
                p: model.NewBoolVar(f'stud_{s["student_id"]}_{c_name}_p{p}')
                for p in all_periods
            } for c_name in s["requests"] if c_name in course_definitions
        } for s in student_requests
    }

    # --- Add all original constraints (copied from build_master_schedule_model) ---
    for student in student_requests:
        s_id = student["student_id"]
        for p in all_periods:
            model.AddAtMostOne(student_in_course_period[s_id][c_name][p] for c_name in student_in_course_period[s_id])

    for c_name, defs in course_definitions.items():
        model.Add(sum(is_section_active[c_name][s][p] for s in range(MAX_SECTIONS_PER_COURSE) for p in all_periods) == defs["total_sections"])

    for c_name in course_definitions:
        for s_idx in range(MAX_SECTIONS_PER_COURSE):
            model.AddAtMostOne(is_section_active[c_name][s_idx][p] for p in all_periods)

    for c_name, defs in course_definitions.items():
        for s_idx in range(MAX_SECTIONS_PER_COURSE):
            students_in_section = [student_assigned_to_section[s["student_id"]][c_name][s_idx] for s in student_requests if c_name in student_assigned_to_section.get(s["student_id"], {})]
            model.Add(sum(students_in_section) <= defs["max_capacity"])

    for dis_course, dis_period in disallowed_course_period_combinations:
        if dis_course in course_definitions:
            for s_idx in range(MAX_SECTIONS_PER_COURSE):
                model.Add(is_section_active[dis_course][s_idx][dis_period] == 0)

    for student in student_requests:
        s_id = student["student_id"]
        for c_name in student["requests"]:
            if c_name in student_assigned_to_section.get(s_id, {}):
                model.AddAtMostOne(student_assigned_to_section[s_id][c_name][s_idx] for s_idx in range(MAX_SECTIONS_PER_COURSE))
    
    for student in student_requests:
        s_id = student["student_id"]
        for c_name in student["requests"]:
            if c_name not in student_in_course_period.get(s_id,{}): continue
            for p in all_periods:
                assignments_in_period = []
                for s_idx in range(MAX_SECTIONS_PER_COURSE):
                    link = model.NewBoolVar(f'link_{s_id}_{c_name}_s{s_idx}_p{p}')
                    model.AddBoolAnd([student_assigned_to_section[s_id][c_name][s_idx], is_section_active[c_name][s_idx][p]]).OnlyEnforceIf(link)
                    assignments_in_period.append(link)
                model.Add(sum(assignments_in_period) >= 1).OnlyEnforceIf(student_in_course_period[s_id][c_name][p])
                model.Add(sum(assignments_in_period) == 0).OnlyEnforceIf(student_in_course_period[s_id][c_name][p].Not())

    for c_name, defs in course_definitions.items():
        if not defs["allow_multiple_sections_in_same_period"]:
            for p in all_periods:
                model.Add(sum(is_section_active[c_name][s_idx][p] for s_idx in range(MAX_SECTIONS_PER_COURSE)) <= 1)

    for c1, c2 in disallowed_course_pairs:
        if c1 in course_definitions and c2 in course_definitions:
            for p in all_periods:
                c1_active_in_p = model.NewBoolVar(f'is_active_{c1}_{p}')
                sum_c1_sections = sum(is_section_active[c1][s_idx][p] for s_idx in range(MAX_SECTIONS_PER_COURSE))
                model.Add(sum_c1_sections > 0).OnlyEnforceIf(c1_active_in_p)
                model.Add(sum_c1_sections == 0).OnlyEnforceIf(c1_active_in_p.Not())

                c2_active_in_p = model.NewBoolVar(f'is_active_{c2}_{p}')
                sum_c2_sections = sum(is_section_active[c2][s_idx][p] for s_idx in range(MAX_SECTIONS_PER_COURSE))
                model.Add(sum_c2_sections > 0).OnlyEnforceIf(c2_active_in_p)
                model.Add(sum_c2_sections == 0).OnlyEnforceIf(c2_active_in_p.Not())
                model.AddAtMostOne([c1_active_in_p, c2_active_in_p])

    # --- Add constraints to fix sections based on current_fixed_offerings ---
    for c_name, periods_data in current_fixed_offerings.items():
        for p, sections_list in periods_data.items():
            for section_info in sections_list:
                s_idx = _get_section_index_from_id(section_info['section_id'])
                if s_idx is None:
                    print(f"Warning: Could not parse section index from {section_info['section_id']}. Skipping fixing.", flush=True)
                    continue

                # Check if this is the section currently being swapped
                is_swapped_section = False
                if section_to_swap_info:
                    if (c_name == section_to_swap_info['course_name'] and
                        s_idx == section_to_swap_info['s_idx'] and
                        (p == section_to_swap_info['old_period'] or p == section_to_swap_info['new_period'])):
                        is_swapped_section = True
                
                # If it's not the section being swapped, fix its active state
                if not is_swapped_section:
                    model.Add(is_section_active[c_name][s_idx][p] == True)
                elif is_swapped_section and p == section_to_swap_info['old_period']:
                    # Explicitly set the swapped section to inactive in its old period
                    model.Add(is_section_active[c_name][s_idx][p] == False)
                elif is_swapped_section and p == section_to_swap_info['new_period']:
                    # Explicitly set the swapped section to active in its new period
                    model.Add(is_section_active[c_name][s_idx][p] == True)

    # If a section was provided to swap, ensure its new period is active and old is inactive
    if section_to_swap_info:
        c_name = section_to_swap_info['course_name']
        s_idx = section_to_swap_info['s_idx']
        old_p = section_to_swap_info['old_period']
        new_p = section_to_swap_info['new_period']
        
        # Ensure the section is active in the new period and inactive in the old period
        model.Add(is_section_active[c_name][s_idx][new_p] == True)
        model.Add(is_section_active[c_name][s_idx][old_p] == False)

        # Also ensure this section is not active in any period other than new_p (already mostly covered by AtMostOne, but explicit for clarity here)
        for p_other in all_periods:
            if p_other != new_p:
                model.Add(is_section_active[c_name][s_idx][p_other] == False)


    # Objective: Maximize the number of fulfilled student requests
    fulfilled_vars = [var for s in student_in_course_period.values() for c in s.values() for var in c.values()]
    model.Maximize(sum(fulfilled_vars))

    return model, {"is_section_active": is_section_active, "student_assigned_to_section": student_assigned_to_section, "student_in_course_period": student_in_course_period}

# --- STAGE 2: Assign Teachers to the Finalized Schedule ---
def assign_teachers(final_course_offerings, teachers, all_periods):
    print("\n--- STAGE 2: Assigning Teachers ---", flush=True)
    if not teachers:
        print("No teacher data loaded. Skipping assignment.", flush=True)
        return final_course_offerings, {}

    teacher_assignments = {t: [] for t in teachers}
    final_teacher_schedules = {t: {p: "FREE" for p in all_periods} for t in teachers}

    for period in all_periods:
        sections_in_this_period = []
        for course_name, sections_in_course in final_course_offerings.items():
            if period in sections_in_course:
                for section_info in sections_in_course[period]:
                    sections_in_this_period.append((course_name, section_info))
        
        for course_name, section_info in sections_in_this_period:
            qualified_teachers = [t for t, info in teachers.items() if course_name in info["subjects_taught"]]
            available_teachers = [t for t in qualified_teachers if final_teacher_schedules[t][period] == "FREE"]

            if not available_teachers:
                section_info['teacher'] = "Unassigned"
            else:
                best_teacher = min(available_teachers, key=lambda t: len(teacher_assignments[t]))
                section_info['teacher'] = best_teacher
                final_teacher_schedules[best_teacher][period] = f"{course_name} ({len(section_info['enrolled'])})"
                teacher_assignments[best_teacher].append(section_info['section_id'])
    
    print("Teacher assignment complete.", flush=True)
    return final_course_offerings, final_teacher_schedules

# --- Results Extraction and Processing ---
def process_results(solver, variables, student_requests, course_definitions, all_periods, teachers):
    is_section_active = variables["is_section_active"]
    student_assigned_to_section = variables["student_assigned_to_section"]
    student_in_course_period = variables["student_in_course_period"]
    
    # Reconstruct course offerings based on the solver's solution
    extracted_course_offerings = {}
    for c_name, defs in course_definitions.items():
        for s_idx in range(MAX_SECTIONS_PER_COURSE):
            for p in all_periods:
                if solver.BooleanValue(is_section_active[c_name][s_idx][p]):
                    if c_name not in extracted_course_offerings: extracted_course_offerings[c_name] = {}
                    if p not in extracted_course_offerings[c_name]: extracted_course_offerings[c_name][p] = []
                    
                    enrolled_students = []
                    for s in student_requests:
                        if c_name in student_assigned_to_section.get(s['student_id'], {}):
                            if solver.BooleanValue(student_assigned_to_section[s['student_id']][c_name][s_idx]):
                                enrolled_students.append(s['student_id'])
                    
                    extracted_course_offerings[c_name][p].append({
                        "capacity": defs["max_capacity"], "enrolled": enrolled_students, "section_id": f"{c_name} (Sec {s_idx+1})"
                    })
    
    final_course_offerings_with_teachers, final_teacher_schedules = assign_teachers(extracted_course_offerings, teachers, all_periods)
    
    final_student_schedules, final_unscheduled_students = {}, []
    
    for student in student_requests:
        s_id, name, requests = student["student_id"], student["name"], student["requests"]
        schedule = {p: "FREE" for p in all_periods}
        unfulfilled = list(requests)
        
        for c_name in requests:
            if c_name in student_in_course_period.get(s_id, {}):
                for p in all_periods:
                    if solver.BooleanValue(student_in_course_period[s_id][c_name][p]):
                        schedule[p] = c_name
                        if c_name in unfulfilled:
                            unfulfilled.remove(c_name)
                        break
                        
        final_student_schedules[s_id] = {"name": name, "schedule": schedule}
        if unfulfilled:
            final_unscheduled_students.append({"student_id": s_id, "name": name, "unfulfilled_requests": unfulfilled})

    return final_student_schedules, final_unscheduled_students, final_course_offerings_with_teachers, final_teacher_schedules

# --- Solution Callback Class for Progress Reporting ---
class SchedulingSolutionCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, student_in_course_period, student_requests):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._student_in_course_period = student_in_course_period
        self._solution_count = 0
        self._start_time = time.time()
        self._total_possible_requests = sum(len(s['requests']) for s in student_requests)

    def OnSolutionCallback(self):
        self._solution_count += 1
        current_time = time.time()
        fulfilled_count = 0
        for s_id, courses in self._student_in_course_period.items():
            for c_name, periods in courses.items():
                for p, var in periods.items():
                    if self.BooleanValue(var):
                        fulfilled_count += 1
        
        if self._total_possible_requests > 0:
            percentage = (fulfilled_count / self._total_possible_requests) * 100
            print(f"    SOLUTION #{self._solution_count}: Found at {current_time - self._start_time:.2f}s, "
                  f"Fulfilled Requests: {percentage:.2f}% ({fulfilled_count}/{self._total_possible_requests}), "
                  f"Objective: {self.ObjectiveValue()}", flush=True)

# --- Output Functions ---
def _print_student_schedules(schedules, periods):
    print("\n--- Student Master Schedules ---", flush=True)
    data = [{"Student ID": sid, "Student Name": s["name"], **s["schedule"]} for sid, s in schedules.items()]
    df = pd.DataFrame(data)[["Student ID", "Student Name"] + periods]
    print(df.to_string(), flush=True)
    df.to_csv(get_unique_filepath("student_master_schedules.csv"), index=False)
    print("\nStudent schedules saved.", flush=True)

def _print_course_enrollment(offerings):
    print("\n--- Course & Period Enrollment ---", flush=True)
    data = []
    for c, p_data in sorted(offerings.items()):
        for p, sections in sorted(p_data.items()):
            for s_info in sections:
                enroll, cap = len(s_info["enrolled"]), s_info["capacity"]
                data.append({
                    "Course Section": s_info["section_id"], "Period": p, "Teacher": s_info.get("teacher", "Unassigned"),
                    "Enrollment": enroll, "Capacity": cap, "Remaining": cap - enroll
                })
    df = pd.DataFrame(data)
    if not df.empty: print(df.to_string(), flush=True)
    df.to_csv(get_unique_filepath("course_enrollment_summary.csv"), index=False)
    print("\nCourse enrollment summary saved.", flush=True)

def _print_master_schedule_grid(course_offerings, all_periods):
    print("\n--- Master Schedule Grid (Courses x Periods) ---", flush=True)
    grid_data = []
    for course_name, periods_data in course_offerings.items():
        for period, sections_list in periods_data.items():
            for section_info in sections_list:
                grid_data.append({ "Course": course_name, "Period": period, "Enrollment": len(section_info["enrolled"]) })
    if grid_data:
        df_grid = pd.DataFrame(grid_data).groupby(['Course', 'Period'])['Enrollment'].sum().unstack(fill_value=0).reindex(columns=all_periods, fill_value=0)
        print(df_grid.sort_index().to_string(), flush=True)
        df_grid.sort_index().to_csv(get_unique_filepath("master_schedule_grid.csv"))
        print(f"\nMaster schedule grid saved.", flush=True)
    else: 
        print("No course sections to display in the grid.", flush=True)
        
def _print_teacher_schedules(schedules, periods):
    print("\n--- Teacher Schedules ---", flush=True)
    df = pd.DataFrame.from_dict(schedules, orient='index', columns=periods).sort_index()
    df.index.name = "Teacher"
    print(df.to_string(), flush=True)
    df.to_csv(get_unique_filepath("teacher_schedules.csv"))
    print("\nTeacher schedules saved.", flush=True)

def _print_unscheduled_students(unscheduled_students):
    print("\n--- Students Unable to Get Full Schedule ---", flush=True)
    if unscheduled_students:
        df = pd.DataFrame(unscheduled_students)
        print(df.to_string(), flush=True)
        df.to_csv(get_unique_filepath("unscheduled_students.csv"), index=False)
        print(f"\nUnscheduled students list saved.", flush=True)
    else: print("All students successfully received a full schedule!", flush=True)

def get_unique_filepath(base_filepath):
    """Generates a unique filepath to prevent overwriting existing files."""
    if not os.path.exists(base_filepath): return base_filepath
    dir, fname = os.path.split(base_filepath)
    name, ext = os.path.splitext(fname)
    i = 1
    while True:
        new_path = os.path.join(dir, f"{name} ({i}){ext}")
        if not os.path.exists(new_path): return new_path
        i += 1

def recalculate_and_print_stats(student_schedules, student_data, unscheduled_students_ref):
    """Recalculates and prints fulfillment stats from the current schedule data."""
    # This function now also rebuilds the list of unscheduled students.
    unscheduled_students_ref.clear()
    total_possible_requests = sum(len(s['requests']) for s in student_data)
    total_fulfilled_requests_count = 0
    fully_scheduled_students_count = 0

    for s_id, schedule_info in student_schedules.items():
        student = next((s for s in student_data if s['student_id'] == s_id), None)
        if not student: continue
        
        student_requests = student['requests']
        courses_in_schedule = schedule_info['schedule'].values()
        
        unfulfilled_for_student = [req for req in student_requests if req not in courses_in_schedule]
        num_fulfilled = len(student_requests) - len(unfulfilled_for_student)
        
        total_fulfilled_requests_count += num_fulfilled
        
        if not unfulfilled_for_student:
            fully_scheduled_students_count += 1
        else:
            unscheduled_students_ref.append({"student_id": s_id, "name": student['name'], "unfulfilled_requests": unfulfilled_for_student})
            
    print("\n--- Current Fulfillment ---")
    if total_possible_requests > 0:
        print(f"Percentage of ALL requested courses fulfilled: {(total_fulfilled_requests_count / total_possible_requests) * 100:.2f}%")
        print(f"Percentage of students who got ALL their requests fulfilled: {(fully_scheduled_students_count / len(student_data)) * 100:.2f}%")

# --- Function to print all available commands ---
def print_available_commands():
    print("\n--- Available Commands ---", flush=True)
    print("Commands: 'assign [StudentID] [Period] [Course]' | 'force_swap [StudentID] [P1] [P2]' | 'show [type]' | 'exit'", flush=True)
    print("          'move_section [CourseName] [SecNum] [OldPeriod] [NewPeriod]' (e.g., 'move_section AlgebraI Sec1 P1 P5')", flush=True)
    print("          'show' types: students, classes, teachers, grid, unfulfilled, percentages, full", flush=True)
    print("          'show commands' (to display this list)", flush=True)

def load_existing_schedule_files(student_schedules_filepath, course_enrollment_filepath, course_definitions, all_periods):
    """
    Loads a previously saved student schedule and course enrollment summary
    to reconstruct the internal schedule data structures.
    """
    loaded_student_schedules = {}
    loaded_course_offerings = {}

    print(f"\nAttempting to load student schedules from '{student_schedules_filepath}'...", flush=True)
    try:
        df_students = pd.read_csv(student_schedules_filepath)
        # Populate loaded_student_schedules
        for _, row in df_students.iterrows():
            s_id = str(row["Student ID"]).strip()
            s_name = str(row["Student Name"]).strip()
            schedule = {p: str(row[p]).strip() for p in all_periods}
            loaded_student_schedules[s_id] = {"name": s_name, "schedule": schedule}
        print(f"Loaded {len(loaded_student_schedules)} student schedules.", flush=True)
    except Exception as e:
        sys.exit(f"ERROR loading existing student schedules: {e}")

    print(f"Attempting to load course enrollment summary from '{course_enrollment_filepath}'...", flush=True)
    try:
        df_enrollment = pd.read_csv(course_enrollment_filepath)
        # First, populate the base course_offerings structure with capacities and teachers
        for _, row in df_enrollment.iterrows():
            course_section_id = str(row["Course Section"]).strip()
            period = str(row["Period"]).strip()
            course_name_match = re.match(r'(.+) \(Sec \d+\)', course_section_id)
            course_name = course_name_match.group(1).strip() if course_name_match else course_section_id.split(' (Sec')[0].strip()
            
            s_idx = _get_section_index_from_id(course_section_id)
            
            if course_name not in course_definitions:
                print(f"Warning: Course '{course_name}' from enrollment summary not found in course definitions. Skipping section '{course_section_id}'.", flush=True)
                continue

            if course_name not in loaded_course_offerings:
                loaded_course_offerings[course_name] = {}
            if period not in loaded_course_offerings[course_name]:
                loaded_course_offerings[course_name][period] = []
            
            loaded_course_offerings[course_name][period].append({
                "capacity": int(row["Capacity"]), 
                "enrolled": [], # Will fill this in the next step based on student schedules
                "section_id": course_section_id,
                "teacher": str(row.get("Teacher", "Unassigned")).strip() 
            })
        print(f"Loaded base course offerings from enrollment summary.", flush=True)

        # Now, iterate through loaded student schedules to populate 'enrolled' lists in course_offerings
        print("Populating enrolled students in sections based on loaded student schedules...", flush=True)
        for s_id, student_info in loaded_student_schedules.items():
            for period, course_name in student_info['schedule'].items():
                if course_name == "FREE": # Skip free periods
                    continue
                
                # Find the correct section to enroll the student into
                # This assumes a student can only be in one section of a course at a time in the loaded schedule.
                # It also assumes the section IDs are consistent.
                section_found = False
                if course_name in loaded_course_offerings and period in loaded_course_offerings[course_name]:
                    for section_data in loaded_course_offerings[course_name][period]:
                        # A simple way to assign: if the section has capacity, add the student.
                        # This might not perfectly recreate which specific section a student was in if multiple sections
                        # of the *same course* existed in the *same period* in the old schedule,
                        # and we don't have that specific student-to-section mapping in student_master_schedules.csv directly.
                        # However, student_master_schedules.csv provides course-period assignment, and course_enrollment_summary.csv
                        # provides all sections for that course-period.
                        
                        # For direct loading, we just need to ensure the student is marked as enrolled in *one* of the sections.
                        # Since `_print_student_schedules` output format only records Course-Period, not Course-Section-Period,
                        # we pick the first available section that matches the course and period.
                        
                        if len(section_data['enrolled']) < section_data['capacity']:
                            section_data['enrolled'].append(s_id)
                            section_found = True
                            break
                if not section_found:
                    print(f"Warning: Student {s_id} requested '{course_name}' in '{period}' but no suitable section with capacity was found in loaded course offerings. This student's enrollment may be inconsistent.", flush=True)

        print("Finished populating student enrollments in sections.", flush=True)

    except Exception as e:
        sys.exit(f"ERROR loading existing course enrollment summary or reconstructing enrollments: {e}")
    
    # Ensure sections are consistently ordered if multiple exist for the same course-period
    for c_name in loaded_course_offerings:
        for p in loaded_course_offerings[c_name]:
            loaded_course_offerings[c_name][p].sort(key=lambda x: _get_section_index_from_id(x['section_id']))

    return loaded_student_schedules, loaded_course_offerings


# --- Main Execution Block ---
if __name__ == "__main__":
    # Define primary input files (always needed for course definitions, student requests, etc.)
    student_requests_file = "Student_Requests.csv"
    disallowed_periods_file = "disallowed_periods.csv"
    course_definitions_file = "course_definitions.csv"
    disallowed_course_pairs_file = "disallowed_course_pairs.csv"
    teacher_data_file = "teacher_data.csv"

    # Define default previous output filenames
    DEFAULT_STUDENT_SCHEDULES_FILE = "student_master_schedules.csv"
    DEFAULT_COURSE_ENROLLMENT_FILE = "course_enrollment_summary.csv"

    # Load base data (definitions, requests, constraints)
    student_data, course_data, periods, disallowed_combos, disallowed_pairs, teacher_data = load_data(
        student_requests_filepath=student_requests_file,
        disallowed_periods_filepath=disallowed_periods_file,
        course_definitions_filepath=course_definitions_file,
        disallowed_course_pairs_filepath=disallowed_course_pairs_file,
        teacher_data_filepath=teacher_data_file
    )

    student_schedules = {}
    unscheduled_students = []
    course_offerings = {}
    teacher_schedules = {}

    load_option = input("\nDo you want to load a previous schedule to continue working (yes/no)? ").strip().lower()

    # Modify the condition to accept 'y' for yes, 'n' for no
    if load_option in ['yes', 'y']:
        # Prompt for filenames, with defaults
        prev_student_schedules_file = input(f"Enter path to previous student_master_schedules.csv (default: {DEFAULT_STUDENT_SCHEDULES_FILE}): ").strip()
        if not prev_student_schedules_file:
            prev_student_schedules_file = DEFAULT_STUDENT_SCHEDULES_FILE

        prev_course_enrollment_file = input(f"Enter path to previous course_enrollment_summary.csv (default: {DEFAULT_COURSE_ENROLLMENT_FILE}): ").strip()
        if not prev_course_enrollment_file:
            prev_course_enrollment_file = DEFAULT_COURSE_ENROLLMENT_FILE
        
        try:
            # Load the previous schedule state directly without involving the solver for initial student assignment
            student_schedules, course_offerings = load_existing_schedule_files(
                prev_student_schedules_file, prev_course_enrollment_file, course_data, periods
            )
            
            # Re-assign teachers based on the loaded schedule (they might have been "Unassigned" or changed)
            if teacher_data:
                course_offerings, teacher_schedules = assign_teachers(course_offerings, teacher_data, periods)
            else:
                print("No teacher data available to re-assign teachers to the loaded schedule.", flush=True)

            print(f"\n--- LOADED SCHEDULE SUMMARY ---")
            recalculate_and_print_stats(student_schedules, student_data, unscheduled_students)
            
            _print_student_schedules(student_schedules, periods)
            _print_unscheduled_students(unscheduled_students)
            _print_course_enrollment(course_offerings)
            _print_master_schedule_grid(course_offerings, periods)
            if teacher_data:
                _print_teacher_schedules(teacher_schedules, periods)

        except Exception as e:
            print(f"Error loading previous schedule files: {e}", flush=True)
            import traceback
            traceback.print_exc(file=sys.stdout)
            sys.exit(1)

    # Modify the else to also handle 'n' for no
    elif load_option in ['no', 'n']:
        print("\n--- STAGE 1: Building Master Schedule Model (from scratch) ---", flush=True)
        model, variables = build_master_schedule_model(student_data, course_data, periods, disallowed_combos, disallowed_pairs)
        
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 600.0
        solver.parameters.log_search_progress = True

        solution_callback = SchedulingSolutionCallback(variables['student_in_course_period'], student_data)

        print("\n--- Solving for Master Schedule (with progress callback) ---", flush=True)
        status = solver.Solve(model, solution_callback)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print("\nMaster schedule solved. Processing results and assigning teachers...", flush=True)
            student_schedules, unscheduled_students, course_offerings, teacher_schedules = \
                process_results(solver, variables, student_data, course_data, periods, teacher_data)
            
            print(f"\n--- FINAL SUMMARY ---")
            recalculate_and_print_stats(student_schedules, student_data, unscheduled_students)
            print(f"Solver statistics: Branches={solver.NumBranches()}, Conflicts={solver.NumConflicts()}, Time={solver.WallTime():.2f}s", flush=True)
            
            _print_student_schedules(student_schedules, periods)
            _print_unscheduled_students(unscheduled_students)
            _print_course_enrollment(course_offerings)
            _print_master_schedule_grid(course_offerings, periods)
            if teacher_data:
                _print_teacher_schedules(teacher_schedules, periods)
        else:
            print("Initial solution: No optimal or feasible solution found.", flush=True)
            print(f"Solver status: {solver.StatusName(status)}", flush=True)
            sys.exit(1)
    else: # Handle invalid input for the initial load option
        print("Invalid option. Please enter 'yes', 'y', 'no', or 'n'. Terminating.", flush=True)
        sys.exit(1)
        
    # --- Interactive Loop for Adjustments ---
    print("\n--- Entering Interactive Schedule Adjustment Mode ---", flush=True)
    print_available_commands() # Call the new function to print commands

    while True:
        try:
            command_line = input("\nEnter command: ").strip()
            if not command_line: continue
    
            parts = command_line.split()
            command = parts[0].lower()
    
            if command in ['exit', 'quit']:
                print("Exiting interactive mode.", flush=True)
                break
            
            elif command == 'assign' or command == 'force_swap':
                if (command == 'assign' and len(parts) == 4) or (command == 'force_swap' and len(parts) == 4):
                    s_id = parts[1]
                    if s_id not in student_schedules:
                        print(f"Error: Student ID '{s_id}' not found.")
                        continue

                    if command == 'assign':
                        p_to, new_course = parts[2], parts[3]
                        if p_to not in periods: print(f"Error: Invalid period '{p_to}'."); continue
                        if new_course not in course_data: print(f"Error: Course '{new_course}' not found."); continue
                        
                        print(f"Assigning {new_course} to {s_id} in {p_to}...")
                        
                        target_sections = course_offerings.get(new_course, {}).get(p_to, [])
                        if not target_sections:
                            print(f"Error: No sections of '{new_course}' are offered in {p_to}. Assignment failed.")
                            continue
                        
                        old_course = student_schedules[s_id]['schedule'][p_to]
                        if old_course != "FREE":
                            for sec in course_offerings.get(old_course, {}).get(p_to, []):
                                if s_id in sec['enrolled']: sec['enrolled'].remove(s_id); break
                        
                        target_sections[0]['enrolled'].append(s_id)
                        student_schedules[s_id]['schedule'][p_to] = new_course

                    elif command == 'force_swap':
                        p1, p2 = parts[2], parts[3]
                        if p1 not in periods or p2 not in periods: print(f"Error: Invalid period specified. Use P1-P8."); continue
                        
                        print(f"Swapping periods for student {s_id} between {p1} and {p2}...")
                        sched = student_schedules[s_id]['schedule']
                        c1_name, c2_name = sched[p1], sched[p2]

                        def find_section(course, period, student_id=None):
                            for sec in course_offerings.get(course, {}).get(period, []):
                                if student_id is None or student_id in sec.get('enrolled', []):
                                    return sec
                            return None

                        old_sec1, old_sec2 = find_section(c1_name, p1, s_id), find_section(c2_name, p2, s_id)
                        new_sec1_target, new_sec2_target = find_section(c2_name, p1), find_section(c1_name, p2)

                        if (c1_name != "FREE" and (not old_sec1 or not new_sec2_target)) or \
                           (c2_name != "FREE" and (not old_sec2 or not new_sec1_target)):
                            print("Error: Could not find necessary sections to perform the swap.")
                            continue

                        sched[p1], sched[p2] = c2_name, c1_name
                        if old_sec1: old_sec1['enrolled'].remove(s_id)
                        if new_sec2_target: new_sec2_target['enrolled'].append(s_id)
                        if old_sec2: old_sec2['enrolled'].remove(s_id)
                        if new_sec1_target: new_sec1_target['enrolled'].append(s_id)

                    # --- Post-Move Updates ---
                    print("\n--- UPDATED SCHEDULE ---")
                    recalculate_and_print_stats(student_schedules, student_data, unscheduled_students)
                    course_offerings, teacher_schedules = assign_teachers(course_offerings, teacher_data, periods)
                    _print_student_schedules(student_schedules, periods)
                    _print_unscheduled_students(unscheduled_students)
                    _print_course_enrollment(course_offerings)
                    _print_master_schedule_grid(course_offerings, periods)
                    if teacher_data: _print_teacher_schedules(teacher_schedules, periods)

                else:
                    print(f"Invalid '{command}' command format. Check help for syntax.")

            elif command == 'move_section': # Changed from 'swap_sections'
                if len(parts) == 5:
                    course_name_to_move = parts[1] # Changed variable name
                    section_num_str = parts[2] # e.g., "Sec1"
                    old_period = parts[3].upper()
                    new_period = parts[4].upper()

                    # Input Validation
                    if course_name_to_move not in course_data: # Changed variable name
                        print(f"Error: Course '{course_name_to_move}' not found in course definitions."); continue # Changed variable name
                    if old_period not in periods or new_period not in periods:
                        print(f"Error: Invalid period specified. Use P1-P8."); continue
                    if old_period == new_period:
                        print(f"Error: Old period and new period are the same. No move needed."); continue # Changed from swap to move
                    
                    s_idx_to_move = _get_section_index_from_id(section_num_str) # Changed variable name
                    if s_idx_to_move is None or s_idx_to_move >= MAX_SECTIONS_PER_COURSE: # Changed variable name
                        print(f"Error: Invalid section number format or out of range: '{section_num_str}'. Expected format like 'Sec1'."); continue
                    
                    # Verify the section exists in the old period
                    section_found_in_old_period = False
                    target_section_info = None
                    if course_name_to_move in course_offerings and old_period in course_offerings[course_name_to_move]: # Changed variable name
                        for sec_info in course_offerings[course_name_to_move][old_period]: # Changed variable name
                            if _get_section_index_from_id(sec_info['section_id']) == s_idx_to_move: # Changed variable name
                                target_section_info = sec_info
                                section_found_in_old_period = True
                                break
                    
                    if not section_found_in_old_period:
                        print(f"Error: Section '{section_num_str}' of '{course_name_to_move}' not found in '{old_period}'. Cannot move."); continue # Changed from swap to move, changed variable name
                    
                    # Check if the target section would conflict with disallowed periods in the new period
                    if (course_name_to_move, new_period) in disallowed_combos: # Changed variable name
                        print(f"Error: Course '{course_name_to_move}' is disallowed in period '{new_period}'. Cannot move."); continue # Changed from swap to move, changed variable name

                    print(f"Attempting to move section {section_num_str} of {course_name_to_move} from {old_period} to {new_period}...", flush=True) # Changed from swap to move, changed variable name

                    # Prepare info for the re-solver
                    section_move_details = { # Changed variable name
                        'course_name': course_name_to_move, # Changed variable name
                        's_idx': s_idx_to_move, # Changed variable name
                        'old_period': old_period,
                        'new_period': new_period
                    }
                    
                    # Rebuild and resolve the model with the new fixed section positions
                    print("\n--- Rebuilding and Re-solving Master Schedule with Fixed Sections ---", flush=True)
                    new_model, new_variables = build_model_for_reschedule(
                        course_offerings, # Pass the current schedule as the base for fixed sections
                        student_data, 
                        course_data, 
                        periods, 
                        disallowed_combos, 
                        disallowed_pairs, 
                        section_to_swap_info=section_move_details # Changed variable name
                    )

                    new_solver = cp_model.CpSolver()
                    new_solver.parameters.max_time_in_seconds = 600.0
                    new_solver.parameters.log_search_progress = True
                    new_solution_callback = SchedulingSolutionCallback(new_variables['student_in_course_period'], student_data)

                    new_status = new_solver.Solve(new_model, new_solution_callback)

                    if new_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                        print("\nReschedule solution found. Processing results...", flush=True)
                        student_schedules, unscheduled_students, course_offerings, teacher_schedules = \
                            process_results(new_solver, new_variables, student_data, course_data, periods, teacher_data)
                        
                        print(f"\n--- UPDATED SCHEDULE AFTER SECTION MOVE ---") # Changed from swap to move
                        recalculate_and_print_stats(student_schedules, student_data, unscheduled_students)
                        print(f"Solver statistics: Branches={new_solver.NumBranches()}, Conflicts={new_solver.NumConflicts()}, Time={new_solver.WallTime():.2f}s", flush=True)
                        
                        _print_student_schedules(student_schedules, periods)
                        _print_unscheduled_students(unscheduled_students)
                        _print_course_enrollment(course_offerings)
                        _print_master_schedule_grid(course_offerings, periods)
                        if teacher_data: _print_teacher_schedules(teacher_schedules, periods)
                    else:
                        print(f"Rescheduling failed: No optimal or feasible solution found for the requested move.", flush=True) # Changed from swap to move
                        print(f"Solver status: {new_solver.StatusName(new_status)}", flush=True)
                        print("The schedule remains unchanged.", flush=True)

                else:
                    print("Invalid 'move_section' command format. Expected: 'move_section [CourseName] [SecNum] [OldPeriod] [NewPeriod]'") # Changed from swap to move
                    print("Example: 'move_section AlgebraI Sec1 P1 P5'", flush=True) # Changed from swap to move

            elif command == 'show':
                if len(parts) > 1:
                    show_type = parts[1].lower()
                    if show_type == 'students': _print_student_schedules(student_schedules, periods)
                    elif show_type == 'classes': _print_course_enrollment(course_offerings)
                    elif show_type == 'teachers': _print_teacher_schedules(teacher_schedules, periods) if teacher_data else print("No teacher data loaded.")
                    elif show_type == 'grid': _print_master_schedule_grid(course_offerings, periods)
                    elif show_type == 'unfulfilled': _print_unscheduled_students(unscheduled_students)
                    elif show_type == 'percentages':
                        recalculate_and_print_stats(student_schedules, student_data, unscheduled_students)
                    elif show_type == 'full':
                         _print_student_schedules(student_schedules, periods)
                         _print_unscheduled_students(unscheduled_students)
                         _print_course_enrollment(course_offerings)
                         _print_master_schedule_grid(course_offerings, periods)
                         if teacher_data: _print_teacher_schedules(teacher_schedules, periods)
                    elif show_type == 'commands': # New command to show available commands
                        print_available_commands()
                    else: print(f"Unknown show command: {show_type}", flush=True)
                else: print("Please specify what to show.", flush=True)
            else:
                print("Unknown command.", flush=True)
        
        except Exception as e:
            print(f"An unexpected error occurred in interactive mode: {e}", flush=True)
            import traceback
            traceback.print_exc(file=sys.stdout)