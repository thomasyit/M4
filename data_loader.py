"""
Data Loader Utility for AI_Bot
Loads OHLCV data from consolidated Binance and Hyperliquid historical data
Optimized for fast local data access instead of API calls
"""

import os
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime

class DataLoader:
    """Load and manage historical OHLCV data"""

    TIMEFRAMES = ['1h', '2h', '4h', '8h', '12h', '1d']
    EXCHANGES = ['binance', 'hyperliquid']

    def __init__(self, base_path: str = None):
        """
        Initialize DataLoader

        Args:
            base_path: Root path to data directory. Defaults to AI_bot folder
        """
        if base_path is None:
            # Use AI_bot folder as base path
            base_path = Path(__file__).parent.absolute()
        else:
            base_path = Path(base_path)

        self.base_path = base_path
        self.binance_path = self.base_path / "binance_historical"
        self.hyperliquid_path = self.base_path / "hyperliquid_historical"

    def load_data(
        self,
        symbol: str,
        timeframe: str,
        exchange: str = 'binance',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load OHLCV data for a symbol and timeframe

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT', 'BTC_USDT')
            timeframe: '1h', '2h', '4h', '8h', '12h', '1d'
            exchange: 'binance' or 'hyperliquid'
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)

        Returns:
            DataFrame with OHLCV data
        """

        if exchange not in self.EXCHANGES:
            raise ValueError(f"Exchange must be one of {self.EXCHANGES}")

        if timeframe not in self.TIMEFRAMES:
            raise ValueError(f"Timeframe must be one of {self.TIMEFRAMES}")

        # Normalize symbol
        symbol = symbol.upper()

        # Determine path
        if exchange == 'binance':
            data_path = self.binance_path / f"ohlcv_{timeframe}"
        else:
            data_path = self.hyperliquid_path / f"ohlcv_{timeframe}"

        # Find matching file
        csv_file = None
        if data_path.exists():
            for file in data_path.glob("*.csv"):
                file_symbol = file.stem.split('_')[0]
                if file_symbol == symbol or symbol in file.name:
                    csv_file = file
                    break

        if not csv_file or not csv_file.exists():
            raise FileNotFoundError(
                f"No data found for {symbol} {timeframe} on {exchange}\n"
                f"Expected in: {data_path}"
            )

        # Load data
        df = pd.read_csv(csv_file)

        # Ensure timestamp column
        if 'timestamp' not in df.columns:
            if 'open_time' in df.columns:
                df['timestamp'] = df['open_time']
            elif 'time' in df.columns:
                df['timestamp'] = df['time']

        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Filter by date range if provided
        if start_date:
            start_dt = pd.to_datetime(start_date)
            # Handle timezone-aware timestamps
            if df['timestamp'].dt.tz is not None:
                start_dt = start_dt.tz_localize('UTC')
            df = df[df['timestamp'] >= start_dt]

        if end_date:
            end_dt = pd.to_datetime(end_date)
            # Handle timezone-aware timestamps
            if df['timestamp'].dt.tz is not None:
                end_dt = end_dt.tz_localize('UTC')
            df = df[df['timestamp'] <= end_dt]

        # Sort by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')

        return df.reset_index(drop=True)

    def list_available_data(self, exchange: str = 'binance') -> dict:
        """
        List all available data

        Args:
            exchange: 'binance' or 'hyperliquid'

        Returns:
            Dictionary with symbol/timeframe combinations and row counts
        """

        if exchange == 'binance':
            base_path = self.binance_path
        else:
            base_path = self.hyperliquid_path

        available = {}

        if not base_path.exists():
            return available

        for timeframe_dir in base_path.glob("ohlcv_*"):
            if timeframe_dir.is_dir():
                timeframe = timeframe_dir.name.replace("ohlcv_", "")
                available[timeframe] = {}

                for csv_file in timeframe_dir.glob("*.csv"):
                    df = pd.read_csv(csv_file, nrows=1)
                    row_count = len(pd.read_csv(csv_file))
                    symbol = csv_file.stem
                    available[timeframe][symbol] = row_count

        return available

    def get_latest_timestamp(
        self,
        symbol: str,
        timeframe: str,
        exchange: str = 'binance'
    ) -> Optional[datetime]:
        """
        Get the latest timestamp for a symbol/timeframe

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            exchange: Exchange name

        Returns:
            Latest timestamp or None if no data
        """
        try:
            df = self.load_data(symbol, timeframe, exchange)
            if 'timestamp' in df.columns:
                return df['timestamp'].max()
            return None
        except:
            return None

    def merge_data(
        self,
        symbol: str,
        timeframe: str,
        new_data: pd.DataFrame,
        exchange: str = 'binance',
        remove_duplicates: bool = True
    ) -> None:
        """
        Merge new data with existing data

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            new_data: New DataFrame to merge
            exchange: Exchange name
            remove_duplicates: Remove duplicate timestamps
        """

        # Determine path
        if exchange == 'binance':
            data_path = self.binance_path / f"ohlcv_{timeframe}"
        else:
            data_path = self.hyperliquid_path / f"ohlcv_{timeframe}"

        data_path.mkdir(parents=True, exist_ok=True)

        # Find or create file
        csv_file = None
        symbol_upper = symbol.upper()

        for file in data_path.glob("*.csv"):
            file_symbol = file.stem.split('_')[0]
            if file_symbol == symbol_upper or symbol_upper in file.name:
                csv_file = file
                break

        if csv_file is None:
            csv_file = data_path / f"{symbol_upper}_{timeframe}.csv"

        # Load existing data if available
        if csv_file.exists():
            existing_df = pd.read_csv(csv_file)
            df = pd.concat([existing_df, new_data], ignore_index=True)
        else:
            df = new_data

        # Remove duplicates
        if remove_duplicates and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.drop_duplicates(subset=['timestamp'], keep='last')
            df = df.sort_values('timestamp')

        # Save
        df.to_csv(csv_file, index=False)
        print(f"✓ Merged data for {symbol}_{timeframe}: {len(df)} rows saved")

    def get_ohlcv_array(
        self,
        symbol: str,
        timeframe: str,
        exchange: str = 'binance',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
        """
        Load OHLCV data and return as separate lists (optimized for trading signals)

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            exchange: Exchange name
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            Tuple of (open_prices, high_prices, low_prices, close_prices, volumes)
        """
        df = self.load_data(symbol, timeframe, exchange, start_date, end_date)

        return (
            df['open'].tolist(),
            df['high'].tolist(),
            df['low'].tolist(),
            df['close'].tolist(),
            df['volume'].tolist()
        )

    def get_latest_candle(
        self,
        symbol: str,
        timeframe: str,
        exchange: str = 'binance'
    ) -> dict:
        """
        Get the latest OHLCV candle for a symbol (useful for signal generation)

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            exchange: Exchange name

        Returns:
            Dictionary with latest candle data
        """
        df = self.load_data(symbol, timeframe, exchange)
        if len(df) == 0:
            return None

        latest = df.iloc[-1]
        return {
            'timestamp': latest.get('timestamp'),
            'open': float(latest['open']),
            'high': float(latest['high']),
            'low': float(latest['low']),
            'close': float(latest['close']),
            'volume': float(latest['volume'])
        }


# Example usage
if __name__ == "__main__":
    loader = DataLoader()

    # List available data
    print("=== Binance Data Available ===")
    binance_data = loader.list_available_data('binance')
    for tf, symbols in sorted(binance_data.items()):
        print(f"\n{tf}:")
        for symbol, count in sorted(symbols.items()):
            print(f"  {symbol}: {count} rows")

    # Load specific data
    print("\n=== Loading Sample Data ===")
    try:
        df = loader.load_data('BTCUSDT', '1h', 'binance')
        print(f"BTC 1h: {len(df)} rows from {df['timestamp'].min()} to {df['timestamp'].max()}")
    except Exception as e:
        print(f"Error: {e}")
