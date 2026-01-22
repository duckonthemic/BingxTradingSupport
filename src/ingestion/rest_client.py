"""
BingX REST API client for fetching market data.
Used for initial data fetching and fallback when WebSocket is unavailable.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

import aiohttp

from ..config import config

logger = logging.getLogger(__name__)


class BingXRestClient:
    """
    Async REST client for BingX API.
    Provides methods to fetch spot market data.
    """
    
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or config.bingx.rest_url
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
    
    async def connect(self):
        """Create HTTP session."""
        if not self._session:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        return self
    
    async def disconnect(self):
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
    
    async def _request(self, method: str, endpoint: str, params: dict = None, **kwargs) -> Optional[Dict[str, Any]]:
        """Make HTTP request with timestamp."""
        # Add timestamp to params
        if params is None:
            params = {}
        params["timestamp"] = int(datetime.now().timestamp() * 1000)
        
        # Build URL with params
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{self.base_url}{endpoint}?{query_string}"
        
        try:
            async with self._session.request(method, url, **kwargs) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"API error {response.status}: {await response.text()}")
                    return None
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None
    
    async def get_all_tickers(self) -> List[Dict[str, Any]]:
        """
        Get 24hr ticker price change statistics for all symbols.
        
        Returns:
            List of ticker data for all trading pairs
        """
        endpoint = "/openApi/spot/v1/ticker/24hr"
        result = await self._request("GET", endpoint)
        
        if result and result.get("code") == 0:
            return result.get("data", [])
        return []
    
    async def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get 24hr ticker for specific symbol.
        
        Args:
            symbol: Trading pair (e.g., "PEPE-USDT")
            
        Returns:
            Ticker data or None
        """
        endpoint = "/openApi/spot/v1/ticker/24hr"
        result = await self._request("GET", endpoint, params={"symbol": symbol})
        
        if result and result.get("code") == 0:
            data = result.get("data", [])
            return data[0] if data else None
        return None
    
    async def get_klines(
        self, 
        symbol: str, 
        interval: str = "1h", 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get candlestick/kline data.
        
        Args:
            symbol: Trading pair
            interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            limit: Number of candles (max 1000)
            
        Returns:
            List of kline data
        """
        endpoint = "/openApi/spot/v1/market/kline"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        result = await self._request("GET", endpoint, params=params)
        
        if result and result.get("code") == 0:
            return result.get("data", [])
        return []
    
    async def get_depth(self, symbol: str, limit: int = 50) -> Optional[Dict[str, Any]]:
        """
        Get orderbook depth.
        
        Args:
            symbol: Trading pair
            limit: Number of price levels
            
        Returns:
            Orderbook data with bids and asks
        """
        endpoint = "/openApi/spot/v1/market/depth"
        params = {"symbol": symbol, "limit": limit}
        result = await self._request("GET", endpoint, params=params)
        
        if result and result.get("code") == 0:
            return result.get("data")
        return None
    
    async def get_funding_rate(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get funding rate for perpetual futures.
        
        Args:
            symbol: Trading pair (e.g., "BTC-USDT")
            
        Returns:
            Dict with lastFundingRate, markPrice, indexPrice, nextFundingTime
        """
        # Convert spot symbol to futures format
        futures_symbol = symbol.replace("-", "-")  # BingX format: BTC-USDT
        
        endpoint = "/openApi/swap/v2/quote/premiumIndex"
        params = {"symbol": futures_symbol}
        result = await self._request("GET", endpoint, params=params)
        
        if result and result.get("code") == 0:
            data = result.get("data", {})
            return {
                "symbol": symbol,
                "funding_rate": float(data.get("lastFundingRate", 0)) * 100,  # Convert to percentage
                "mark_price": float(data.get("markPrice", 0)),
                "index_price": float(data.get("indexPrice", 0)),
                "next_funding_time": data.get("nextFundingTime", 0)
            }
        return None
    
    # ==================== FUTURES MARKET ENDPOINTS ====================
    
    async def get_futures_tickers(self) -> List[Dict[str, Any]]:
        """
        Get all perpetual futures tickers.
        
        Returns:
            List of futures ticker data
        """
        endpoint = "/openApi/swap/v2/quote/ticker"
        result = await self._request("GET", endpoint)
        
        if result and result.get("code") == 0:
            return result.get("data", [])
        return []
    
    async def get_futures_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get single futures ticker.
        
        Args:
            symbol: Trading pair (e.g., "BTC-USDT")
        """
        endpoint = "/openApi/swap/v2/quote/ticker"
        params = {"symbol": symbol}
        result = await self._request("GET", endpoint, params=params)
        
        if result and result.get("code") == 0:
            data = result.get("data", [])
            return data[0] if isinstance(data, list) else data
        return None
    
    async def get_futures_klines(
        self, 
        symbol: str, 
        interval: str = "15m", 
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List:
        """
        Get futures candlestick/kline data.
        
        Args:
            symbol: Trading pair (e.g., "BTC-USDT")
            interval: Kline interval (1m, 5m, 15m, 30m, 1h, 4h, 1d)
            limit: Number of candles (max 1440)
            start_time: Start timestamp in milliseconds (optional)
            end_time: End timestamp in milliseconds (optional)
            
        Returns:
            List of kline data [time, open, high, low, close, volume, ...]
        """
        endpoint = "/openApi/swap/v3/quote/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
            
        result = await self._request("GET", endpoint, params=params)
        
        if result and result.get("code") == 0:
            return result.get("data", [])
        return []
    
    async def get_futures_klines_historical(
        self,
        symbol: str,
        interval: str = "15m",
        start_date: datetime = None,
        end_date: datetime = None,
        max_candles: int = 5000
    ) -> List[Dict]:
        """
        Fetch historical futures klines with pagination for backtesting.
        
        Args:
            symbol: Trading pair (e.g., "BTC-USDT")
            interval: Kline interval (1m, 5m, 15m, 30m, 1h, 4h, 1d)
            start_date: Start datetime
            end_date: End datetime
            max_candles: Maximum candles to fetch (safety limit)
            
        Returns:
            List of candle dicts with timestamp, open, high, low, close, volume
        """
        all_candles = []
        
        # Default to last 7 days if not specified
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=7)
            
        current_start = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)
        
        # Interval to ms mapping
        interval_ms = {
            "1m": 60000, "5m": 300000, "15m": 900000,
            "30m": 1800000, "1h": 3600000, "4h": 14400000, "1d": 86400000
        }
        step_ms = interval_ms.get(interval, 900000)
        
        while current_start < end_ts and len(all_candles) < max_candles:
            # Fetch batch (max 1440 per request)
            batch_end = min(current_start + 1440 * step_ms, end_ts)
            
            raw_candles = await self.get_futures_klines(
                symbol=symbol,
                interval=interval,
                limit=1440,
                start_time=current_start,
                end_time=batch_end
            )
            
            if not raw_candles:
                break
                
            # Parse candles to dict format
            for c in raw_candles:
                if isinstance(c, dict):
                    all_candles.append({
                        "timestamp": datetime.fromtimestamp(c.get("time", 0) / 1000),
                        "open": float(c.get("open", 0)),
                        "high": float(c.get("high", 0)),
                        "low": float(c.get("low", 0)),
                        "close": float(c.get("close", 0)),
                        "volume": float(c.get("volume", 0))
                    })
                elif isinstance(c, list) and len(c) >= 6:
                    # Array format: [time, open, high, low, close, vol, ...]
                    all_candles.append({
                        "timestamp": datetime.fromtimestamp(c[0] / 1000),
                        "open": float(c[1]),
                        "high": float(c[2]),
                        "low": float(c[3]),
                        "close": float(c[4]),
                        "volume": float(c[5])
                    })
            
            # Move to next batch
            if raw_candles:
                last_ts = raw_candles[-1][0] if isinstance(raw_candles[-1], list) else raw_candles[-1].get("time", 0)
                current_start = last_ts + step_ms
            else:
                break
                
            await asyncio.sleep(0.2)  # Rate limit
        
        logger.info(f"📊 Fetched {len(all_candles)} historical candles for {symbol}")
        return all_candles
    
    async def get_futures_depth(self, symbol: str, limit: int = 50) -> Optional[Dict[str, Any]]:
        """
        Get futures orderbook depth.
        
        Args:
            symbol: Trading pair
            limit: Number of price levels
        """
        endpoint = "/openApi/swap/v2/quote/depth"
        params = {"symbol": symbol, "limit": limit}
        result = await self._request("GET", endpoint, params=params)
        
        if result and result.get("code") == 0:
            return result.get("data")
        return None
    
    async def get_open_interest(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get open interest for a futures symbol.
        
        Args:
            symbol: Trading pair
            
        Returns:
            Dict with openInterest and symbol
        """
        endpoint = "/openApi/swap/v2/quote/openInterest"
        params = {"symbol": symbol}
        result = await self._request("GET", endpoint, params=params)
        
        if result and result.get("code") == 0:
            return result.get("data")
        return None


# ==================== Test Function ====================

async def test_rest_client():
    """Test REST API client."""
    print("=" * 50)
    print("Testing BingX REST API Client...")
    print("=" * 50)
    
    async with BingXRestClient() as client:
        # Test get all tickers
        print("\n📊 Fetching all tickers...")
        tickers = await client.get_all_tickers()
        
        if tickers:
            print(f"✅ Got {len(tickers)} tickers")
            
            # Show 5 sample USDT pairs
            usdt_tickers = [t for t in tickers if t.get("symbol", "").endswith("-USDT")][:5]
            print(f"\n📈 Sample USDT pairs ({len(usdt_tickers)}):")
            for t in usdt_tickers:
                symbol = t.get("symbol", "Unknown")
                price = float(t.get("lastPrice", 0))
                volume = float(t.get("quoteVolume", 0))
                # priceChangePercent may have % sign
                change_str = str(t.get("priceChangePercent", "0")).replace("%", "")
                change = float(change_str) if change_str else 0
                print(f"   {symbol}: ${price:.6g} | Vol: ${volume:,.0f} | {change:+.2f}%")
        else:
            print("❌ Failed to get tickers")
            return
        
        # Test get specific ticker
        print("\n📌 Fetching BTC-USDT ticker...")
        btc = await client.get_ticker("BTC-USDT")
        if btc:
            print(f"✅ BTC-USDT: ${float(btc['lastPrice']):,.2f}")
        
        # Test get klines
        print("\n📊 Fetching BTC-USDT klines (1h, last 5)...")
        klines = await client.get_klines("BTC-USDT", "1h", 5)
        if klines:
            print(f"✅ Got {len(klines)} klines")
            for k in klines[-3:]:
                # Klines might be array [time, open, high, low, close, vol] or dict
                if isinstance(k, dict):
                    print(f"   Open: ${float(k['open']):,.2f} | Close: ${float(k['close']):,.2f}")
                elif isinstance(k, list) and len(k) >= 5:
                    print(f"   Open: ${float(k[1]):,.2f} | Close: ${float(k[4]):,.2f}")
        
        # Test get depth
        print("\n📚 Fetching BTC-USDT orderbook...")
        depth = await client.get_depth("BTC-USDT", 5)
        if depth:
            bids = depth.get('bids', [])
            asks = depth.get('asks', [])
            if bids:
                print(f"✅ Best bid: ${float(bids[0][0]):,.2f}")
            if asks:
                print(f"✅ Best ask: ${float(asks[0][0]):,.2f}")
    
    print("\n✅ All REST API tests completed!")


if __name__ == "__main__":
    asyncio.run(test_rest_client())
