from app import app, db
from models import Stock

# with app.app_context():
#     stocks_data = [
#         {'ticker': 'AAPL', 'shares': 50, 'average_cost': 185.30},
#         {'ticker': 'MSFT', 'shares': 30, 'average_cost': 405.20},
#         {'ticker': 'VOO', 'shares': 25, 'average_cost': 418.50},
#     ]
    
#     for stock_data in stocks_data:
#         stock = Stock(user_id=2, **stock_data)
#         db.session.add(stock)
    
#     db.session.commit()
#     print("Test stocks added!")

with app.app_context():
    stocks = Stock.query.filter_by(user_id=2).all()
    for s in stocks:
        print(f"{s.ticker}: {s.shares} @ ${s.average_cost}")