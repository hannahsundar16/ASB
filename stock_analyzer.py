import pandas as pd

class StockAnalyzerTool:
    name = "stock_analyzer"
    description = (
        "Analyze stock transactions. "
        "Supports: 'symbol:<SYMBOL>', 'risky', 'recent'."
    )

    def __init__(self, csv_path: str = "synthetic_stock_dataset_20250704_164724.csv"):
        self.data = pd.read_csv(csv_path)

    def _run(self, query: str) -> str:
        try:
            if query.startswith("symbol:"):
                symbol = query.split("symbol:")[1].strip()
                result = self.data[self.data["symbol"].str.upper() == symbol.upper()]
            elif query.startswith("risky"):
                result = self.data[self.data["risk_tolerance"].str.lower() == "aggressive"]
            elif query.startswith("recent"):
                result = self.data.sort_values(by="trade_date", ascending=False).head(5)
            else:
                return "Invalid query format. Try 'symbol:AAPL', 'risky', or 'recent'."
           
            return result.to_string(index=False)
        except Exception as e:
            return f"Error: {e}"