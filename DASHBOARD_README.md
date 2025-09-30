# 📊 Iris Query Analytics Dashboard

A comprehensive dashboard system for managing access to the `/query` endpoint, tracking user queries, monitoring performance, and analyzing LLM reasoning steps.

## 🌟 Features

### Query Analytics & Monitoring
- **Real-time Query Tracking**: Monitor all queries to the `/query` endpoint
- **Response Time Analysis**: Track and analyze response times
- **Success Rate Monitoring**: Monitor query success/failure rates
- **User Activity Tracking**: Track queries per user and session
- **Reasoning Step Capture**: Store and analyze LLM reasoning processes

### User Management
- **User Registration**: Create and manage user accounts
- **Access Control**: Grant/revoke access to specific endpoints
- **Rate Limiting**: Set custom rate limits per user (hourly/daily)
- **Bulk Operations**: Perform bulk user management operations
- **User Analytics**: View detailed user statistics and behavior

### Access Control & Security
- **Rate Limiting**: Configurable rate limits per user/session
- **IP Blocking**: Block specific IP addresses
- **User Blocking**: Temporarily or permanently block users
- **Endpoint-specific Access**: Control access to specific endpoints
- **Real-time Monitoring**: Monitor rate limit violations

### Dashboard Interface
- **Overview Dashboard**: Key metrics and real-time statistics
- **User Management**: Create, edit, and manage users
- **Query Logs**: View detailed query logs with filtering
- **Analytics Charts**: Visual analytics with time-series data
- **Access Control Panel**: Manage permissions and rate limits

## 🚀 Quick Start

### 1. Setup Database and Sample Data

```bash
# Run the setup script to create tables and sample data
python setup_dashboard.py
```

### 2. Start the Backend

```bash
# Start FastAPI backend with analytics middleware
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Start the Dashboard

```bash
# Start Streamlit dashboard
streamlit run app/dashboard_app.py --server.port 8502
```

### 4. Access the Interfaces

- **Dashboard UI**: http://localhost:8502
- **API Documentation**: http://localhost:8000/docs
- **Original Chat Interface**: http://localhost:8501

## 📁 File Structure

```
app/
├── middleware/
│   ├── __init__.py
│   ├── analytics_middleware.py    # Query tracking middleware
│   └── rate_limiter.py           # Rate limiting and access control
├── routers/
│   ├── chatbot.py               # Enhanced with tracking
│   ├── dashboard.py             # Dashboard API endpoints
│   └── documents.py
├── schemas/
│   ├── analytics.py             # Analytics data models
│   ├── chat.py
│   └── document.py
├── services/
│   ├── analytics_service.py     # Analytics business logic
│   ├── chatbot_service.py
│   └── document_service.py
├── dashboard_app.py             # Streamlit dashboard interface
├── database.py                  # Enhanced with analytics tables
└── main.py                      # Enhanced with middleware

setup_dashboard.py               # Database setup script
DASHBOARD_README.md             # This file
```

## 🗄️ Database Schema

### New Tables Added

#### `users`
- User account information
- Daily query limits
- Admin privileges
- Activity tracking

#### `query_logs`
- Complete query tracking
- Response times and token usage
- Reasoning steps (JSON)
- Error tracking

#### `user_sessions`
- Session management
- Query counts per session
- Activity timestamps

#### `access_control`
- Per-user access rules
- Rate limiting configuration
- Endpoint-specific permissions

## 🔧 API Endpoints

### Dashboard Management
- `GET /dashboard/summary` - Dashboard overview
- `GET /dashboard/analytics` - Query analytics over time
- `GET /dashboard/health` - Health check

### User Management
- `GET /dashboard/users` - List all users
- `POST /dashboard/users` - Create new user
- `PUT /dashboard/users/{user_id}` - Update user
- `DELETE /dashboard/users/{user_id}` - Delete user
- `POST /dashboard/users/bulk` - Bulk user operations

### Query Logs
- `GET /dashboard/queries` - Get query logs (with filtering)
- `GET /dashboard/queries/{query_id}` - Get detailed query log

### Access Control
- `GET /dashboard/access-control/{user_id}` - Get access rules
- `POST /dashboard/access-control` - Set access rules
- `PUT /dashboard/access-control/{user_id}` - Update access rules

### Rate Limiting
- `GET /dashboard/rate-limit/{user_id}` - Check rate limit status

## 📊 Analytics Features

### Query Tracking
Every query to `/query` is automatically tracked with:
- User ID and session ID
- Query text and response
- Response time in milliseconds
- Success/error status
- IP address and user agent
- LLM reasoning steps (if available)
- Token usage and model information

### Reasoning Step Analysis
The system captures detailed reasoning steps from the LLM:
- **Thoughts**: Internal reasoning processes
- **Actions**: Tool usage and function calls
- **Observations**: Results from tool executions
- **Final Answer**: The complete response

### Performance Metrics
- Average response times
- Query volume trends
- Success rates over time
- User activity patterns
- Peak usage analysis

## 🛡️ Security Features

### Rate Limiting
- **Hourly Limits**: Configurable per user (default: 60/hour)
- **Daily Limits**: Configurable per user (default: 1000/day)
- **Sliding Window**: Uses sliding window algorithm for accurate limiting
- **Graceful Degradation**: Returns 429 status with retry information

### Access Control
- **User-level Blocking**: Temporarily or permanently block users
- **IP-level Blocking**: Block specific IP addresses
- **Endpoint-specific Access**: Control access to individual endpoints
- **Admin Privileges**: Special permissions for admin users

### Monitoring
- **Real-time Alerts**: Monitor for suspicious activity
- **Audit Logs**: Complete audit trail of all queries
- **Error Tracking**: Detailed error logging and analysis

## 🎛️ Configuration

### Environment Variables
Add these to your `.env` file:

```env
# Existing variables...

# Analytics Configuration
ANALYTICS_ENABLED=true
RATE_LIMITING_ENABLED=true
DEFAULT_HOURLY_LIMIT=60
DEFAULT_DAILY_LIMIT=1000

# Dashboard Configuration
DASHBOARD_ADMIN_USER=admin_user
DASHBOARD_SECRET_KEY=your_dashboard_secret_key
```

### Rate Limit Defaults
- **Regular Users**: 60 queries/hour, 1000 queries/day
- **Admin Users**: 200 queries/hour, 2000 queries/day
- **Test Users**: 30 queries/hour, 200 queries/day

## 🔍 Usage Examples

### Creating a User via API
```python
import requests

user_data = {
    "user_id": "new_user",
    "username": "New User",
    "email": "user@example.com",
    "daily_query_limit": 100,
    "is_admin": False
}

response = requests.post(
    "http://localhost:8000/dashboard/users",
    json=user_data
)
```

### Setting Rate Limits
```python
access_data = {
    "user_id": "new_user",
    "endpoint": "/query",
    "is_allowed": True,
    "rate_limit_per_hour": 30,
    "rate_limit_per_day": 300
}

response = requests.post(
    "http://localhost:8000/dashboard/access-control",
    json=access_data
)
```

### Checking Rate Limit Status
```python
response = requests.get(
    "http://localhost:8000/dashboard/rate-limit/new_user"
)
status = response.json()
print(f"Queries this hour: {status['queries_this_hour']}")
```

## 🚨 Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Ensure SQLite database is accessible
   - Run `python setup_dashboard.py` to initialize

2. **Middleware Not Working**
   - Check that middleware is properly registered in `main.py`
   - Verify import paths are correct

3. **Dashboard Not Loading**
   - Ensure FastAPI backend is running on port 8000
   - Check Streamlit is running on port 8502
   - Verify API endpoints are accessible

4. **Rate Limiting Not Working**
   - Check that analytics middleware is enabled
   - Verify user exists in database
   - Check access control settings

### Debug Mode
Enable debug logging by setting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## 📝 License

This dashboard system is part of the Iris project and follows the same license terms.
