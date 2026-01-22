"""
Pre-filter for BingX ticker data.
Filters coins based on volume and trading pair criteria.
"""

import logging
from dataclasses import dataclass
from typing import Optional
from enum import Enum

from ..config import config

logger = logging.getLogger(__name__)


class CoinCategory(Enum):
    """Classification of coins by volume."""
    LOW_CAP = "LOW_CAP"      # $50k - $5M
    MID_CAP = "MID_CAP"      # $5M - $50M
    EXCLUDED = "EXCLUDED"    # Outside range


@dataclass
class FilteredTicker:
    """
    Represents a ticker that passed pre-filter.
    Contains normalized data for further processing.
    """
    symbol: str              # e.g., "PEPE-USDT"
    base_asset: str          # e.g., "PEPE"
    price: float             # Current price
    volume_24h: float        # 24h volume in USDT
    price_change_pct: float  # 24h price change %
    high_24h: float          # 24h high
    low_24h: float           # 24h low
    category: CoinCategory   # LOW_CAP or MID_CAP
    timestamp: int           # Unix timestamp
    
    @property
    def price_range_pct(self) -> float:
        """Calculate 24h price range as percentage."""
        if self.low_24h > 0:
            return ((self.high_24h - self.low_24h) / self.low_24h) * 100
        return 0.0
    
    @property
    def is_low_cap(self) -> bool:
        return self.category == CoinCategory.LOW_CAP
    
    @property
    def is_mid_cap(self) -> bool:
        return self.category == CoinCategory.MID_CAP


class PreFilter:
    """
    Pre-filter for ticker data.
    Applies rules to filter out unsuitable coins early.
    """
    
    def __init__(self):
        self.min_volume = config.filter.min_volume_24h
        self.max_volume = config.filter.max_volume_24h
        self.mid_cap_threshold = config.filter.mid_cap_threshold
        
        # Stats
        self._total_received = 0
        self._total_passed = 0
        self._rejected_by_pair = 0
        self._rejected_by_volume_low = 0
        self._rejected_by_volume_high = 0
        self._rejected_by_invalid = 0
        
        logger.info(f"PreFilter initialized: volume ${self.min_volume:,.0f} - ${self.max_volume:,.0f}")
    
    def filter(self, ticker: dict) -> Optional[FilteredTicker]:
        """
        Apply pre-filter rules to a single ticker.
        
        Args:
            ticker: Raw ticker data from BingX WebSocket
            
        Returns:
            FilteredTicker if passed, None if rejected
        """
        self._total_received += 1
        
        # === RULE 1: Must be USDT pair ===
        symbol = ticker.get("symbol", "")
        if not symbol.endswith("-USDT"):
            self._rejected_by_pair += 1
            return None
        
        # === RULE 2: Extract and validate price ===
        try:
            price = float(ticker.get("lastPrice") or ticker.get("c") or 0)
            if price <= 0:
                self._rejected_by_invalid += 1
                return None
        except (ValueError, TypeError):
            self._rejected_by_invalid += 1
            return None
        
        # === RULE 3: Volume in acceptable range ===
        try:
            # BingX uses 'quoteVolume' or 'qv' for USDT volume
            volume_24h = float(ticker.get("quoteVolume") or ticker.get("qv") or 0)
        except (ValueError, TypeError):
            self._rejected_by_invalid += 1
            return None
        
        if volume_24h < self.min_volume:
            self._rejected_by_volume_low += 1
            return None
        
        if volume_24h > self.max_volume:
            self._rejected_by_volume_high += 1
            return None
        
        # === All rules passed ===
        self._total_passed += 1
        
        # Determine category
        category = CoinCategory.MID_CAP if volume_24h >= self.mid_cap_threshold else CoinCategory.LOW_CAP
        
        # Extract base asset (remove -USDT)
        base_asset = symbol.replace("-USDT", "")
        
        # Parse other fields with defaults
        try:
            price_change_pct = float(ticker.get("priceChangePercent") or ticker.get("p") or 0)
            high_24h = float(ticker.get("highPrice") or ticker.get("h") or price)
            low_24h = float(ticker.get("lowPrice") or ticker.get("l") or price)
            timestamp = int(ticker.get("timestamp") or ticker.get("E") or 0)
        except (ValueError, TypeError):
            price_change_pct = 0
            high_24h = price
            low_24h = price
            timestamp = 0
        
        return FilteredTicker(
            symbol=symbol,
            base_asset=base_asset,
            price=price,
            volume_24h=volume_24h,
            price_change_pct=price_change_pct,
            high_24h=high_24h,
            low_24h=low_24h,
            category=category,
            timestamp=timestamp
        )
    
    def filter_batch(self, tickers: list[dict]) -> list[FilteredTicker]:
        """
        Filter a batch of tickers.
        
        Args:
            tickers: List of raw ticker data
            
        Returns:
            List of FilteredTicker objects that passed
        """
        return [ft for t in tickers if (ft := self.filter(t)) is not None]
    
    @property
    def stats(self) -> dict:
        """Get filter statistics."""
        pass_rate = (self._total_passed / self._total_received * 100) if self._total_received > 0 else 0
        return {
            "total_received": self._total_received,
            "total_passed": self._total_passed,
            "pass_rate": f"{pass_rate:.1f}%",
            "rejected": {
                "non_usdt_pair": self._rejected_by_pair,
                "volume_too_low": self._rejected_by_volume_low,
                "volume_too_high": self._rejected_by_volume_high,
                "invalid_data": self._rejected_by_invalid
            }
        }
    
    def reset_stats(self):
        """Reset statistics counters."""
        self._total_received = 0
        self._total_passed = 0
        self._rejected_by_pair = 0
        self._rejected_by_volume_low = 0
        self._rejected_by_volume_high = 0
        self._rejected_by_invalid = 0


# ==================== Test Function ====================

def test_prefilter():
    """Test pre-filter with sample data."""
    print("=" * 50)
    print("Testing PreFilter...")
    print("=" * 50)
    
    pf = PreFilter()
    
    # Test cases
    test_tickers = [
        # Valid LOW_CAP
        {
            "symbol": "PEPE-USDT",
            "lastPrice": "0.000012",
            "quoteVolume": "1000000",
            "priceChangePercent": "5.5",
            "highPrice": "0.000013",
            "lowPrice": "0.000011"
        },
        # Valid MID_CAP
        {
            "symbol": "ARB-USDT",
            "lastPrice": "1.25",
            "quoteVolume": "10000000",
            "priceChangePercent": "-2.1"
        },
        # Rejected - not USDT
        {
            "symbol": "BTC-ETH",
            "lastPrice": "20.5",
            "quoteVolume": "5000000"
        },
        # Rejected - volume too low
        {
            "symbol": "TINY-USDT",
            "lastPrice": "0.001",
            "quoteVolume": "10000"
        },
        # Rejected - volume too high (large cap)
        {
            "symbol": "BTC-USDT",
            "lastPrice": "50000",
            "quoteVolume": "999999999"
        },
        # Invalid data
        {
            "symbol": "BROKEN-USDT",
            "lastPrice": "invalid",
            "quoteVolume": "1000000"
        }
    ]
    
    print("\nTest Results:")
    print("-" * 40)
    
    for ticker in test_tickers:
        result = pf.filter(ticker)
        symbol = ticker.get("symbol", "Unknown")
        if result:
            print(f"✅ {symbol}: PASSED ({result.category.value})")
            print(f"   Price: ${result.price}, Vol: ${result.volume_24h:,.0f}")
        else:
            print(f"❌ {symbol}: REJECTED")
    
    print("\n" + "-" * 40)
    print("Statistics:")
    for key, value in pf.stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    - {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # Assertions
    assert pf._total_passed == 2, f"Expected 2 passed, got {pf._total_passed}"
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_prefilter()
