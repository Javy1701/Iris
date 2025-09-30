#!/usr/bin/env python3
"""
Setup script for Iris Query Analytics Dashboard

This script initializes the database with the new analytics tables
and creates sample data for testing.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import Base, engine, SessionLocal
from app.services.analytics_service import AnalyticsService
from app.schemas.analytics import UserCreate, AccessControlCreate
from datetime import datetime, timezone

def setup_database():
    """Create all database tables."""
    print("Creating database tables...")
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")
    except Exception as e:
        print(f"❌ Error creating database tables: {e}")
        return False
    return True

def create_sample_data():
    """Create sample users and access control data."""
    print("Creating sample data...")
    
    db = SessionLocal()
    try:
        analytics_service = AnalyticsService(db)
        
        # Create sample users
        sample_users = [
            {
                "user_id": "admin_user",
                "username": "Administrator",
                "email": "admin@iris.com",
                "is_admin": True,
                "daily_query_limit": 1000
            },
            {
                "user_id": "test_user_1",
                "username": "Test User 1",
                "email": "test1@iris.com",
                "is_admin": False,
                "daily_query_limit": 100
            },
            {
                "user_id": "test_user_2",
                "username": "Test User 2",
                "email": "test2@iris.com",
                "is_admin": False,
                "daily_query_limit": 50
            },
            {
                "user_id": "crystal1701",
                "username": "Crystal (Default)",
                "email": "crystal@iris.com",
                "is_admin": False,
                "daily_query_limit": 200
            }
        ]
        
        for user_data in sample_users:
            try:
                # Check if user already exists
                existing_user = analytics_service.get_user(user_data["user_id"])
                if not existing_user:
                    user_create = UserCreate(**user_data)
                    created_user = analytics_service.create_user(user_create)
                    print(f"✅ Created user: {created_user.user_id}")
                else:
                    print(f"ℹ️  User already exists: {user_data['user_id']}")
            except Exception as e:
                print(f"❌ Error creating user {user_data['user_id']}: {e}")
        
        # Create sample access control rules
        access_rules = [
            {
                "user_id": "admin_user",
                "endpoint": "/query",
                "is_allowed": True,
                "rate_limit_per_hour": 200,
                "rate_limit_per_day": 2000
            },
            {
                "user_id": "test_user_1",
                "endpoint": "/query",
                "is_allowed": True,
                "rate_limit_per_hour": 60,
                "rate_limit_per_day": 500
            },
            {
                "user_id": "test_user_2",
                "endpoint": "/query",
                "is_allowed": True,
                "rate_limit_per_hour": 30,
                "rate_limit_per_day": 200
            }
        ]
        
        for rule_data in access_rules:
            try:
                access_create = AccessControlCreate(**rule_data)
                created_rule = analytics_service.set_access_control(rule_data["user_id"], access_create)
                print(f"✅ Created access rule for: {created_rule.user_id}")
            except Exception as e:
                print(f"❌ Error creating access rule for {rule_data['user_id']}: {e}")
        
        print("✅ Sample data created successfully")
        
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")
    finally:
        db.close()

def main():
    """Main setup function."""
    print("🎨 Iris Query Analytics Dashboard Setup")
    print("=" * 50)
    
    # Setup database
    if not setup_database():
        print("❌ Database setup failed. Exiting.")
        sys.exit(1)
    
    # Create sample data
    create_sample_data()
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Start the FastAPI backend:")
    print("   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
    print("\n2. Start the dashboard:")
    print("   streamlit run app/dashboard_app.py --server.port 8502")
    print("\n3. Access the dashboard at: http://localhost:8502")
    print("4. Access the API docs at: http://localhost:8000/docs")
    print("\nSample users created:")
    print("- admin_user (Administrator)")
    print("- test_user_1, test_user_2 (Test users)")
    print("- crystal1701 (Default user)")

if __name__ == "__main__":
    main()
