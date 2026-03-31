from sqlalchemy import create_engine, MetaData, Table, select
import models
from sqlalchemy.orm import Session
import os

engine = create_engine('sqlite:///instance/portfolio.db')

session = Session(engine)

# session.query(models.User).filter(models.User.email == "baluna26@g.holycross.edu").delete(synchronize_session=False)
stmt = select(models.User)

for user in session.scalars(stmt):
   print(user.email)
   
session.expire_all()