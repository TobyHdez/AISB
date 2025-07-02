# scheduler.py
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
    Loads all necessary data from file-like objects.
    """
    student_requests, course_definitions, disallowed_course_period_combinations, disallowed_course_pairs, teachers = [], {}, [], [], {}

    # --- Load Student Requests ---
    try:
        df_requests = pd.read_csv(student_requests_filepath)
        for _, row in df_requests.iterrows():
            requests = [str(row[col]).strip() for col in df_requests.columns if col.startswith('course_') and pd.notna(row[col])]
            student_requests.append({"student_id": str(row["student_id"]).strip(), "name": str(row["name"]).strip(), "requests": requests})
    except Exception as e:
        raise ValueError(f"ERROR loading student requests: {e}")

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
    except Exception as e:
        raise ValueError(f"ERROR loading course definitions: {e}")

    # --- Load Disallowed Course-Period Combinations ---
    try:
        df_disallowed = pd.read_csv(disallowed_periods_filepath)
        for _, row in df_disallowed.iterrows():
            if pd.notna(row.get("course_name")) and pd.notna(row.get("period")):
                disallowed_course_period_combinations.append((str(row["course_name"]).strip(), str(row["period"]).strip()))
    except Exception as e:
        raise ValueError(f"ERROR loading disallowed periods: {e}")
        
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
        except Exception as e:
            raise ValueError(f"ERROR loading disallowed course pairs: {e}")

    # --- Load Teacher Data ---
    if teacher_data_filepath:
        try:
            df_teachers = pd.read_csv(teacher_data_filepath)
            for _, row in df_teachers.iterrows():
                teacher_name = str(row["teacher_name"]).strip()
                subjects = [s.strip() for s in str(row["subjects_taught"]).split('|') if s.strip()]
                teachers[teacher_name] = { "subjects_taught": [s for s in subjects if s in course_definitions] }
        except Exception as e:
           raise ValueError(f"ERROR loading teacher data: {e}")

    return student_requests, course_definitions, all_periods, disallowed_course_period_combinations, disallowed_course_pairs, teachers

def _get_section_index_from_id(section_id_str):
    """Extracts 0-indexed section number from a string like 'Course Name (Sec N)' or 'SecN'."""
    match = re.search(r'\(Sec (\d+)\)', section_id_str)
    if match: return int(match.group(1)) - 1
    match = re.search(r'Sec(\d+)', section_id_str)
    if match: return int(match.group(1)) - 1
    return None

def build_master_schedule_model(student_requests, course_definitions, all_periods, disallowed_course_period_combinations, disallowed_course_pairs):
    model = cp_model.CpModel()

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

    fulfilled_vars = [var for s in student_in_course_period.values() for c in s.values() for var in c.values()]
    model.Maximize(sum(fulfilled_vars))

    return model, {"is_section_active": is_section_active, "student_assigned_to_section": student_assigned_to_section, "student_in_course_period": student_in_course_period}

def assign_teachers(final_course_offerings, teachers, all_periods):
    if not teachers: return final_course_offerings, {}

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
    
    return final_course_offerings, final_teacher_schedules

def process_results(solver, variables, student_requests, course_definitions, all_periods, teachers):
    is_section_active = variables["is_section_active"]
    student_assigned_to_section = variables["student_assigned_to_section"]
    student_in_course_period = variables["student_in_course_period"]
    
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

# --- Functions to create DataFrames for Streamlit ---
def create_student_schedules_df(schedules, periods):
    data = [{"Student ID": sid, "Student Name": s["name"], **s["schedule"]} for sid, s in schedules.items()]
    df = pd.DataFrame(data)
    if not df.empty:
        return df[["Student ID", "Student Name"] + periods]
    return pd.DataFrame(columns=["Student ID", "Student Name"] + periods)


def create_course_enrollment_df(offerings):
    data = []
    for c, p_data in sorted(offerings.items()):
        for p, sections in sorted(p_data.items()):
            for s_info in sections:
                enroll, cap = len(s_info["enrolled"]), s_info["capacity"]
                data.append({
                    "Course Section": s_info["section_id"], "Period": p, "Teacher": s_info.get("teacher", "Unassigned"),
                    "Enrollment": enroll, "Capacity": cap, "Remaining": cap - enroll
                })
    if not data:
        return pd.DataFrame(columns=["Course Section", "Period", "Teacher", "Enrollment", "Capacity", "Remaining"])
    return pd.DataFrame(data)

def create_master_schedule_grid_df(course_offerings, all_periods):
    grid_data = []
    for course_name, periods_data in course_offerings.items():
        for period, sections_list in periods_data.items():
            for section_info in sections_list:
                grid_data.append({ "Course": course_name, "Period": period, "Enrollment": len(section_info["enrolled"]) })
    if grid_data:
        df_grid = pd.DataFrame(grid_data).groupby(['Course', 'Period'])['Enrollment'].sum().unstack(fill_value=0).reindex(columns=all_periods, fill_value=0)
        return df_grid.sort_index()
    return pd.DataFrame()
        
def create_teacher_schedules_df(schedules, periods):
    if not schedules:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(schedules, orient='index', columns=periods).sort_index()
    df.index.name = "Teacher"
    return df

def create_unscheduled_students_df(unscheduled_students):
    if unscheduled_students:
        return pd.DataFrame(unscheduled_students)
    return pd.DataFrame(columns=["student_id", "name", "unfulfilled_requests"])

def get_fulfillment_stats(student_schedules, student_data):
    total_possible_requests = sum(len(s['requests']) for s in student_data)
    if total_possible_requests == 0:
        return {"total_fulfilled": 0, "total_possible": 0, "percentage_fulfilled": 0, "fully_scheduled_students": 0, "total_students": len(student_data), "percentage_fully_scheduled": 0}

    total_fulfilled_requests_count = 0
    fully_scheduled_students_count = 0

    for s_id, schedule_info in student_schedules.items():
        student = next((s for s in student_data if s['student_id'] == s_id), None)
        if not student: continue
        
        student_requests = student['requests']
        courses_in_schedule = [c for c in schedule_info['schedule'].values() if c != "FREE"]
        
        unfulfilled_for_student = [req for req in student_requests if req not in courses_in_schedule]
        num_fulfilled = len(student_requests) - len(unfulfilled_for_student)
        
        total_fulfilled_requests_count += num_fulfilled
        
        if not unfulfilled_for_student:
            fully_scheduled_students_count += 1
            
    return {
        "total_fulfilled": total_fulfilled_requests_count,
        "total_possible": total_possible_requests,
        "percentage_fulfilled": (total_fulfilled_requests_count / total_possible_requests) * 100,
        "fully_scheduled_students": fully_scheduled_students_count,
        "total_students": len(student_data),
        "percentage_fully_scheduled": (fully_scheduled_students_count / len(student_data)) * 100 if student_data else 0
    }