import os
import time
from datetime import datetime, date

def delete_today_files(directory='training'):
    today = date.today()
    count = 0

    print(f"Checking files in {directory}...")

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        if os.path.isfile(filepath):
            file_time = datetime.fromtimestamp(os.path.getctime(filepath)).date()

            if file_time == today:
                try:
                    os.remove(filepath)
                    count += 1
                    if count % 1000 == 0:  # Progress update every 1000 files
                        print(f"Deleted {count} files...")
                except Exception as e:
                    print(f"Error deleting {filename}: {e}")

    print(f"\nOperation completed. Deleted {count} files created today.")

if __name__ == '__main__':
    # Add a safety prompt
    response = input("This will delete all files created TODAY in the training directory. Continue? (y/n): ")
    if response.lower() == 'y':
        delete_today_files()
    else:
        print("Operation cancelled.")