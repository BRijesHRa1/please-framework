"""
Database Cleaning Script
Removes all data from the database and resets it
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from database import reset_db, init_db


def clean_database(auto_confirm: bool = False):
    """
    Clean and reset the entire database
    WARNING: This will delete ALL data!
    
    Args:
        auto_confirm: Skip confirmation prompt if True
    """
    
    print("="*80)
    print("DATABASE CLEANING SCRIPT")
    print("="*80)
    
    # Confirm action
    print("\n⚠️  WARNING: This will DELETE ALL DATA in the database!")
    print("   - All projects will be removed")
    print("   - All cycles will be removed")
    print("   - All agent outputs will be removed")
    print("   - All tasks will be removed")
    print("   - All artifacts will be removed")
    print("   - All reports will be removed")
    
    if not auto_confirm:
        response = input("\n🔴 Are you sure you want to continue? (yes/no): ").strip().lower()
        
        if response != 'yes':
            print("\n❌ Database cleaning cancelled.")
            print("="*80)
            return
    else:
        print("\n🔄 Auto-confirmed via --yes flag")
    
    # Reset database
    print("\n🔄 Resetting database...")
    try:
        reset_db()
        print("\n✅ Database has been cleaned and reset successfully!")
        print("   - All tables dropped")
        print("   - All tables recreated")
        print("   - Database is now empty and ready for use")
        
    except Exception as e:
        print(f"\n❌ Error cleaning database: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Check for --yes flag
    auto_confirm = '--yes' in sys.argv or '-y' in sys.argv
    clean_database(auto_confirm=auto_confirm)

