# app.py (Final Version with Download Buttons)
import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model
import io
import time

# --- Page Configuration (must be the first Streamlit command) ---
st.set_page_config(
    layout="wide",
    page_title="School Master Scheduler",
    page_icon="🏫"
)

# --- Import your refactored functions from scheduler.py ---
from scheduler import (
    load_data,
    build_master_schedule_model,
    process_results,
    create_student_schedules_df,
    create_course_enrollment_df,
    create_master_schedule_grid_df,
    create_teacher_schedules_df,
    create_unscheduled_students_df,
    get_fulfillment_stats
)

# --- NEW: Helper function to convert DataFrame to Excel in memory ---
def to_excel(df: pd.DataFrame):
    output = io.BytesIO()
    # Use the openpyxl engine to write the excel file
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Schedule')
    # Get the bytes from the buffer
    processed_data = output.getvalue()
    return processed_data

st.title("🏫 School Master Scheduler")
st.write("Upload your data files to generate an optimized master schedule for your school.")

# --- File Uploaders ---
st.header("1. Upload Data Files")

with st.expander("Upload Required and Optional CSV Files", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        student_requests_file = st.file_uploader("Student Requests*", type="csv")
        course_definitions_file = st.file_uploader("Course Definitions*", type="csv")
        disallowed_periods_file = st.file_uploader("Disallowed Periods*", type="csv")
    with col2:
        disallowed_course_pairs_file = st.file_uploader("Disallowed Course Pairs (optional)", type="csv")
        teacher_data_file = st.file_uploader("Teacher Data (optional)", type="csv")

# --- Run Scheduler ---
st.header("2. Generate Schedule")
if st.button("🚀 Run Scheduler"):
    if not (student_requests_file and course_definitions_file and disallowed_periods_file):
        st.error("Please upload all required files marked with an asterisk (*).")
    else:
        try:
            with st.spinner("Loading data and building model... Please wait."):
                def to_buffer(uploaded_file):
                    return io.StringIO(uploaded_file.getvalue().decode("utf-8"))

                student_data, course_data, periods, disallowed_combos, disallowed_pairs, teacher_data = load_data(
                    to_buffer(student_requests_file),
                    to_buffer(disallowed_periods_file),
                    to_buffer(course_definitions_file),
                    to_buffer(disallowed_course_pairs_file) if disallowed_course_pairs_file else None,
                    to_buffer(teacher_data_file) if teacher_data_file else None
                )
                model, variables = build_master_schedule_model(student_data, course_data, periods, disallowed_combos, disallowed_pairs)
            
            st.info("Model built. Starting the solver... This may take several minutes.")
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 600.0
            
            start_time = time.time()
            status = solver.Solve(model)
            end_time = time.time()

            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                st.success(f"✅ Feasible solution found in {end_time - start_time:.2f} seconds!")

                student_schedules, unscheduled_students, course_offerings, teacher_schedules = \
                    process_results(solver, variables, student_data, course_data, periods, teacher_data)
                
                # --- Create DataFrames from results ---
                student_schedules_df = create_student_schedules_df(student_schedules, periods)
                unscheduled_students_df = create_unscheduled_students_df(unscheduled_students)
                course_enrollment_df = create_course_enrollment_df(course_offerings)
                master_grid_df = create_master_schedule_grid_df(course_offerings, periods)
                teacher_schedules_df = create_teacher_schedules_df(teacher_schedules, periods)

                stats = get_fulfillment_stats(student_schedules, student_data)
                st.header("📊 Fulfillment Statistics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Requests Fulfilled", f"{stats['percentage_fulfilled']:.2f}%", f"{stats['total_fulfilled']} / {stats['total_possible']}")
                col2.metric("Students Fully Scheduled", f"{stats['percentage_fully_scheduled']:.2f}%", f"{stats['fully_scheduled_students']} / {stats['total_students']}")
                col3.metric("Solver Wall Time", f"{solver.WallTime():.2f}s")
                
                st.header("📋 Generated Schedules")
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Student Schedules", "Unscheduled Students", "Course Enrollment", "Master Grid", "Teacher Schedules"])

                with tab1:
                    st.subheader("Student Master Schedules")
                    st.dataframe(student_schedules_df)
                    st.download_button(
                        label="📥 Download as Excel",
                        data=to_excel(student_schedules_df),
                        file_name="student_schedules.xlsx",
                        mime="application/vnd.ms-excel"
                    )
                
                with tab2:
                    st.subheader("Students with Unfulfilled Requests")
                    if not unscheduled_students_df.empty:
                        st.dataframe(unscheduled_students_df)
                        st.download_button(
                            label="📥 Download as Excel",
                            data=to_excel(unscheduled_students_df),
                            file_name="unscheduled_students.xlsx",
                            mime="application/vnd.ms-excel"
                        )
                    else:
                        st.write("🎉 All students received a full schedule!")

                with tab3:
                    st.subheader("Course & Period Enrollment")
                    st.dataframe(course_enrollment_df)
                    st.download_button(
                        label="📥 Download as Excel",
                        data=to_excel(course_enrollment_df),
                        file_name="course_enrollment.xlsx",
                        mime="application/vnd.ms-excel"
                    )

                with tab4:
                    st.subheader("Master Schedule Grid (Courses x Periods)")
                    st.dataframe(master_grid_df)
                    st.download_button(
                        label="📥 Download as Excel",
                        data=to_excel(master_grid_df),
                        file_name="master_schedule_grid.xlsx",
                        mime="application/vnd.ms-excel"
                    )
                
                with tab5:
                    st.subheader("Teacher Schedules")
                    if teacher_data:
                        st.dataframe(teacher_schedules_df)
                        st.download_button(
                            label="📥 Download as Excel",
                            data=to_excel(teacher_schedules_df),
                            file_name="teacher_schedules.xlsx",
                            mime="application/vnd.ms-excel"
                        )
                    else:
                        st.info("No teacher data was uploaded, so teacher schedules were not generated.")
            else:
                st.error("❌ No optimal or feasible solution was found within the time limit.")
                st.write(f"Solver status: {solver.StatusName(status)}")
        except Exception as e:
            st.error(f"An error occurred during the scheduling process: {e}")