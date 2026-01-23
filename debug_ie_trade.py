"""
Debug IE Trade Scanner - Check why MSS detected but no Ready setups
"""
import asyncio
import sys
sys.path.insert(0, '.')

from src.ie_trade.scanner import IEScanner, ScanPhase
from src.ie_trade.config import IETradeConfig, DEFAULT_CONFIG
from src.ie_trade.bias_manager import BiasManager, DailyBias
from src.ingestion.rest_client import BingXRestClient

async def debug_ie_scanner():
    """Run IE Scanner in debug mode"""
    
    print("=" * 60)
    print("IE TRADE SCANNER DEBUG")
    print("=" * 60)
    
    # Create REST client with context manager
    async with BingXRestClient() as rest_client:
        # Create scanner with custom fetch function
        async def fetch_candles(symbol: str, timeframe: str, limit: int):
            return await rest_client.get_klines(symbol, timeframe, limit)
        
        async def send_alert(message: str):
            print(f"\n{'='*40}\nALERT:\n{message}\n{'='*40}\n")
        
        config = DEFAULT_CONFIG
        scanner = IEScanner(
            config=config,
            fetch_candles=fetch_candles,
            send_alert=send_alert
        )
        
        # Set bias to SHORT
        scanner.bias_manager.set_bias(DailyBias.SHORT, "debug_user")
        print(f"✅ Bias set to: SHORT")
        
        # Run one scan cycle with debug output
        from datetime import datetime
        vn_hour = (datetime.utcnow().hour + 7) % 24
        is_kz, kz_name = config.is_kill_zone(vn_hour)
        print(f"✅ Current time (VN): {vn_hour}:00")
        print(f"✅ In Kill Zone: {is_kz} ({kz_name})")
        print(f"✅ Scanning {len(config.TOP_COINS)} coins...")
        print()
        
        # Scan each coin with detailed output
        direction = "SHORT"
        
        for symbol in config.TOP_COINS[:5]:  # Only first 5 to be faster
            print(f"\n--- {symbol} ---")
            
            # Get candles
            h1_candles = await fetch_candles(symbol, "1h", 100)
            if not h1_candles:
                print(f"  ❌ No H1 candles")
                continue
            
            from src.ie_trade.fvg_detector import candles_from_api_data
            candles = candles_from_api_data(h1_candles)
            current_price = candles[-1].close
            print(f"  Price: ${current_price:.4f}")
            
            # Check FVG
            fvg = scanner.fvg_detector.find_best_fvg(
                candles=candles,
                symbol=symbol,
                timeframe="1h",
                direction=direction,
                current_price=current_price
            )
            
            if fvg:
                print(f"  ✅ FVG: {fvg.bottom:.4f} - {fvg.top:.4f} (Zone: {fvg.zone_type.value if fvg.zone_type else 'N/A'})")
                print(f"     Valid: {fvg.is_valid}, Fresh: {fvg.is_fresh(24)}")
                print(f"     Price in gap: {fvg.is_price_in_gap(current_price)}")
                
                if fvg.is_price_in_gap(current_price):
                    print(f"  ✅ Price IN FVG ZONE! Checking M5 MSS...")
                    
                    # Get M5 candles
                    m5_data = await fetch_candles(symbol, "5m", 100)
                    if m5_data:
                        m5_candles = candles_from_api_data(m5_data)
                        
                        mss = scanner.mss_detector.detect_mss(
                            candles=m5_candles,
                            symbol=symbol,
                            direction=direction,
                            in_fvg_zone=True
                        )
                        
                        if mss:
                            print(f"  ✅ MSS Detected! Displacement: {mss.displacement_body_ratio:.0%}")
                            print(f"     Valid: {mss.is_valid}")
                            print(f"     Entry FVG: {mss.entry_fvg is not None}")
                            
                            # Calculate setup
                            setup = scanner.entry_calculator.calculate_setup(
                                symbol=symbol,
                                direction=direction,
                                current_price=current_price,
                                h1_fvg=fvg,
                                m5_fvg=mss.entry_fvg,
                                mss=mss,
                                h1_candles=candles,
                                m5_candles=m5_candles,
                                kill_zone=kz_name,
                                daily_bias=direction
                            )
                            
                            if setup:
                                print(f"  ✅ Setup created!")
                                print(f"     Entry: ${setup.entry_price:.4f}")
                                print(f"     SL: ${setup.stop_loss:.4f}")
                                print(f"     TP1: ${setup.take_profit_1:.4f}")
                                print(f"     R:R: {setup.rr_ratio_1:.2f}")
                                print(f"     Valid: {setup.is_valid}")
                                if not setup.is_valid:
                                    print(f"     ❌ Reason: {setup.invalidation_reason}")
                            else:
                                print(f"  ❌ Setup is None (calculate_setup failed)")
                        else:
                            print(f"  ⏳ No MSS detected yet")
            else:
                print(f"  ⏳ No FVG found")
    
    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(debug_ie_scanner())
