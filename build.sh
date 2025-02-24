#!/usr/bin/env bash
# exit on error
set -o errexit

# Set the Django settings module before importing Django
export DJANGO_SETTINGS_MODULE="testerally_be.settings"

# Ensure that the virtual environment is activated
source .venv/bin/activate

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Run migrations
python manage.py migrate

# Create superuser programmatically with predefined credentials
python -c "
import os
import django

# Set the DJANGO_SETTINGS_MODULE environment variable
os.environ['DJANGO_SETTINGS_MODULE'] = 'testerally_be.settings'

# Setup Django
django.setup()

# Import User model after setup
from django.contrib.auth.models import User

# Superuser credentials
username = 'admin'
email = 'admin@example.com'
password = 'admin'

# Check if the superuser already exists, if not, create it
if not User.objects.filter(username=username).exists():
    User.objects.create_superuser(username=username, email=email, password=password)
    print(f'Superuser {username} with email {email} created successfully.')
else:
    print(f'Superuser {username} already exists.')
"
