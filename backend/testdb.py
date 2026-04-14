from app import app, db
from models import User

with app.app_context():
    users = User.query.all()
    if not users:
        print('No users found')
    for user in users:
        print(f"{user.id}: {user.email} ({user.display_name})")