"""
Fix FALSE text cells to proper checkboxes (batch with delay)
Also supports fixing during /endall command
"""
import gspread
from google.oauth2.service_account import Credentials
import time
import sys

# Google Sheets setup
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SHEET_ID = "1yqYjPWOZGTD6DMClrCvv_Gt5ZhftuUxuat1leqWTsIw"

def get_client():
    """Get authenticated gspread client"""
    creds = Credentials.from_service_account_file(
        "credentials.json", scopes=SCOPES
    )
    return gspread.authorize(creds)

def find_false_cells(sheet):
    """Find all cells with FALSE text in column N"""
    all_values = sheet.get_all_values()
    false_rows = []
    
    for i, row in enumerate(all_values[5:], start=6):  # Skip header
        if len(row) >= 14:
            end_trade = row[13]  # Column N (0-indexed = 13)
            # Check if it's FALSE text (not a proper checkbox)
            if end_trade and end_trade.upper() == "FALSE":
                false_rows.append(i)
    
    return false_rows

def fix_checkboxes_batch(sheet, rows, delay_seconds=3):
    """Fix FALSE text to checkboxes with delay between batches"""
    if not rows:
        print("✅ No FALSE text found!")
        return 0
    
    print(f"Found {len(rows)} cells with FALSE text")
    
    fixed = 0
    batch_size = 5  # Process 5 rows at a time
    
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        
        # Create batch update for checkbox validation
        requests = []
        for row in batch:
            requests.append({
                "repeatCell": {
                    "range": {
                        "sheetId": sheet.id,
                        "startRowIndex": row - 1,  # 0-indexed
                        "endRowIndex": row,
                        "startColumnIndex": 13,  # Column N
                        "endColumnIndex": 14
                    },
                    "cell": {
                        "dataValidation": {
                            "condition": {
                                "type": "BOOLEAN"
                            }
                        }
                    },
                    "fields": "dataValidation"
                }
            })
        
        try:
            # Apply checkbox validation
            sheet.spreadsheet.batch_update({"requests": requests})
            
            # Clear the FALSE text (set to empty = unchecked)
            for row in batch:
                sheet.update_acell(f'N{row}', '')
                fixed += 1
            
            print(f"  Fixed rows {batch} ({fixed}/{len(rows)})")
            
            # Delay before next batch
            if i + batch_size < len(rows):
                print(f"  Waiting {delay_seconds}s to avoid quota...")
                time.sleep(delay_seconds)
                
        except Exception as e:
            print(f"  Error fixing batch {batch}: {e}")
            if "429" in str(e) or "Quota" in str(e):
                print("  Quota exceeded, waiting 60s...")
                time.sleep(60)
                # Retry this batch
                try:
                    sheet.spreadsheet.batch_update({"requests": requests})
                    for row in batch:
                        sheet.update_acell(f'N{row}', '')
                        fixed += 1
                    print(f"  Retry successful: {batch}")
                except Exception as e2:
                    print(f"  Retry failed: {e2}")
    
    return fixed

def main():
    print("=" * 50)
    print("🔧 Fix FALSE Checkboxes Tool")
    print("=" * 50)
    
    client = get_client()
    spreadsheet = client.open_by_key(SHEET_ID)
    sheet = spreadsheet.get_worksheet(0)
    
    false_rows = find_false_cells(sheet)
    
    if not false_rows:
        print("✅ No FALSE text found, all checkboxes OK!")
        return
    
    print(f"\n📋 Found {len(false_rows)} cells with FALSE text:")
    print(f"   Rows: {false_rows[:10]}{'...' if len(false_rows) > 10 else ''}")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--fix":
        print("\n🔧 Fixing checkboxes...")
        fixed = fix_checkboxes_batch(sheet, false_rows, delay_seconds=3)
        print(f"\n✅ Fixed {fixed}/{len(false_rows)} checkboxes")
    else:
        print("\n⚠️ Run with --fix to fix the checkboxes")
        print("   python fix_false_checkboxes.py --fix")

if __name__ == "__main__":
    main()
