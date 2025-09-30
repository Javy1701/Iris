import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

# Configure Streamlit page
st.set_page_config(
    page_title="Iris Query Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Base URL
API_BASE_URL = "http://localhost:8000"

def make_api_request(endpoint, method="GET", data=None):
    """Make API request with error handling."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        elif method == "PUT":
            response = requests.put(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None

def main():
    st.title("🎨 Iris Query Analytics Dashboard")
    st.markdown("Monitor and manage access to the /query endpoint")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Users", "Query Logs", "Analytics", "Access Control", "Settings"]
    )
    
    if page == "Overview":
        show_overview()
    elif page == "Users":
        show_users()
    elif page == "Query Logs":
        show_query_logs()
    elif page == "Analytics":
        show_analytics()
    elif page == "Access Control":
        show_access_control()
    elif page == "Settings":
        show_settings()

def show_overview():
    """Display dashboard overview."""
    st.header("📈 Dashboard Overview")
    
    # Get dashboard summary
    summary = make_api_request("/dashboard/summary")
    if not summary:
        st.error("Failed to load dashboard summary")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Users", summary["total_users"])
    
    with col2:
        st.metric("Active Users Today", summary["active_users_today"])
    
    with col3:
        st.metric("Queries Today", summary["total_queries_today"])
    
    with col4:
        st.metric("Avg Response Time", f"{summary['average_response_time']:.1f}ms")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Users")
        if summary["top_users"]:
            top_users_df = pd.DataFrame(summary["top_users"])
            fig = px.bar(
                top_users_df, 
                x="user_id", 
                y="total_queries",
                title="Top Users by Query Count"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No user data available")
    
    with col2:
        st.subheader("Recent Activity")
        if summary["recent_queries"]:
            recent_df = pd.DataFrame(summary["recent_queries"])
            recent_df["created_at"] = pd.to_datetime(recent_df["created_at"])
            
            # Status distribution
            status_counts = recent_df["status"].value_counts()
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Query Status Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No recent queries")

def show_users():
    """Display user management interface."""
    st.header("👥 User Management")
    
    # User creation form
    with st.expander("Create New User"):
        with st.form("create_user"):
            user_id = st.text_input("User ID*")
            username = st.text_input("Username")
            email = st.text_input("Email")
            daily_limit = st.number_input("Daily Query Limit", min_value=1, value=100)
            is_admin = st.checkbox("Admin Privileges")
            
            if st.form_submit_button("Create User"):
                if user_id:
                    user_data = {
                        "user_id": user_id,
                        "username": username or None,
                        "email": email or None,
                        "daily_query_limit": daily_limit,
                        "is_admin": is_admin
                    }
                    result = make_api_request("/dashboard/users", "POST", user_data)
                    if result:
                        st.success(f"User {user_id} created successfully!")
                        st.rerun()
                else:
                    st.error("User ID is required")
    
    # Users list
    users = make_api_request("/dashboard/users?limit=1000")
    if users:
        st.subheader("All Users")
        
        # Convert to DataFrame
        users_df = pd.DataFrame(users)
        users_df["created_at"] = pd.to_datetime(users_df["created_at"])
        users_df["last_active"] = pd.to_datetime(users_df["last_active"])
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            status_filter = st.selectbox("Status", ["All", "Active", "Inactive"])
        with col2:
            admin_filter = st.selectbox("Admin", ["All", "Admin", "Regular"])
        with col3:
            search_term = st.text_input("Search User ID")
        
        # Apply filters
        filtered_df = users_df.copy()
        if status_filter == "Active":
            filtered_df = filtered_df[filtered_df["is_active"] == True]
        elif status_filter == "Inactive":
            filtered_df = filtered_df[filtered_df["is_active"] == False]
        
        if admin_filter == "Admin":
            filtered_df = filtered_df[filtered_df["is_admin"] == True]
        elif admin_filter == "Regular":
            filtered_df = filtered_df[filtered_df["is_admin"] == False]
        
        if search_term:
            filtered_df = filtered_df[filtered_df["user_id"].str.contains(search_term, case=False)]
        
        # Display users
        st.dataframe(
            filtered_df[["user_id", "username", "email", "is_active", "is_admin", "daily_query_limit", "last_active"]],
            use_container_width=True
        )
        
        # Bulk operations
        st.subheader("Bulk Operations")
        selected_users = st.multiselect("Select Users", filtered_df["user_id"].tolist())
        
        if selected_users:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Activate Selected"):
                    bulk_data = {"user_ids": selected_users, "operation": "activate"}
                    result = make_api_request("/dashboard/users/bulk", "POST", bulk_data)
                    if result:
                        st.success("Users activated successfully!")
                        st.rerun()
            
            with col2:
                if st.button("Deactivate Selected"):
                    bulk_data = {"user_ids": selected_users, "operation": "deactivate"}
                    result = make_api_request("/dashboard/users/bulk", "POST", bulk_data)
                    if result:
                        st.success("Users deactivated successfully!")
                        st.rerun()
            
            with col3:
                new_limit = st.number_input("Set Query Limit", min_value=1, value=100, key="bulk_limit")
                if st.button("Update Limits"):
                    bulk_data = {
                        "user_ids": selected_users, 
                        "operation": "set_limit",
                        "parameters": {"daily_query_limit": new_limit}
                    }
                    result = make_api_request("/dashboard/users/bulk", "POST", bulk_data)
                    if result:
                        st.success("Query limits updated successfully!")
                        st.rerun()

def show_query_logs():
    """Display query logs interface."""
    st.header("📝 Query Logs")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        user_filter = st.text_input("User ID Filter")
    
    with col2:
        status_filter = st.selectbox("Status", ["All", "success", "error", "rate_limited"])
    
    with col3:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
    
    with col4:
        end_date = st.date_input("End Date", datetime.now())
    
    # Build query parameters
    params = f"?limit=100"
    if user_filter:
        params += f"&user_id={user_filter}"
    if status_filter != "All":
        params += f"&status={status_filter}"
    if start_date:
        params += f"&start_date={start_date}T00:00:00"
    if end_date:
        params += f"&end_date={end_date}T23:59:59"
    
    # Get query logs
    logs = make_api_request(f"/dashboard/queries{params}")
    if logs:
        logs_df = pd.DataFrame(logs)
        logs_df["created_at"] = pd.to_datetime(logs_df["created_at"])
        
        # Display logs
        st.dataframe(
            logs_df[["id", "user_id", "query_text", "status", "response_time_ms", "created_at"]],
            use_container_width=True
        )
        
        # Detailed view
        if not logs_df.empty:
            selected_log_id = st.selectbox("View Details", logs_df["id"].tolist())
            if selected_log_id:
                detailed_log = make_api_request(f"/dashboard/queries/{selected_log_id}")
                if detailed_log:
                    st.subheader("Query Details")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.text_area("Query", detailed_log["query_text"], height=100)
                    with col2:
                        st.text_area("Response", detailed_log["response_text"] or "No response", height=100)
                    
                    if detailed_log.get("reasoning_breakdown"):
                        st.subheader("Reasoning Steps")
                        for step in detailed_log["reasoning_breakdown"]:
                            with st.expander(f"Step {step['step_number']}: {step['step_type']}"):
                                st.write(step["content"])
                                if step.get("tool_used"):
                                    st.write(f"Tool used: {step['tool_used']}")

def show_analytics():
    """Display analytics charts."""
    st.header("📊 Analytics")
    
    # Time range selector
    days = st.selectbox("Time Range", [7, 14, 30, 60, 90], index=2)
    
    # Get analytics data
    analytics = make_api_request(f"/dashboard/analytics?days={days}")
    if analytics:
        analytics_df = pd.DataFrame(analytics)
        analytics_df["date"] = pd.to_datetime(analytics_df["date"])
        
        # Query volume over time
        fig1 = px.line(
            analytics_df, 
            x="date", 
            y="query_count",
            title="Query Volume Over Time"
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Response time trends
        fig2 = px.line(
            analytics_df, 
            x="date", 
            y="average_response_time",
            title="Average Response Time Trends"
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Success rate
        fig3 = px.line(
            analytics_df, 
            x="date", 
            y="success_rate",
            title="Success Rate Over Time"
        )
        st.plotly_chart(fig3, use_container_width=True)

def show_access_control():
    """Display access control interface."""
    st.header("🔒 Access Control")
    
    # Rate limit checker
    st.subheader("Check Rate Limits")
    user_id = st.text_input("User ID to Check")
    if user_id and st.button("Check Rate Limits"):
        rate_status = make_api_request(f"/dashboard/rate-limit/{user_id}")
        if rate_status:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Queries This Hour", rate_status["queries_this_hour"])
            with col2:
                st.metric("Queries Today", rate_status["queries_today"])
            with col3:
                status_color = "red" if rate_status["is_rate_limited"] else "green"
                st.markdown(f"**Status:** <span style='color:{status_color}'>{'Rate Limited' if rate_status['is_rate_limited'] else 'OK'}</span>", unsafe_allow_html=True)

def show_settings():
    """Display settings interface."""
    st.header("⚙️ Settings")
    
    st.subheader("System Information")
    health = make_api_request("/dashboard/health")
    if health:
        st.json(health)
    
    st.subheader("API Endpoints")
    st.markdown("""
    - **Dashboard Summary:** `GET /dashboard/summary`
    - **Users:** `GET /dashboard/users`
    - **Query Logs:** `GET /dashboard/queries`
    - **Analytics:** `GET /dashboard/analytics`
    - **Rate Limits:** `GET /dashboard/rate-limit/{user_id}`
    """)

if __name__ == "__main__":
    main()
