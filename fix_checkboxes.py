"""
Fix FALSE text to proper checkboxes in Google Sheet.
"""

import gspread
from google.oauth2.service_account import Credentials
import time

SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
SPREADSHEET_ID = '1yqYjPWOZGTD6DMClrCvv_Gt5ZhftuUxuat1leqWTsIw'

def fix_checkboxes():
    # Connect
    creds = Credentials.from_service_account_file('credentials.json', scopes=SCOPES)
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key(SPREADSHEET_ID)
    
    sheet = spreadsheet.sheet1
    sheet_id = sheet.id
    
    print(f"📊 Checking sheet: {spreadsheet.title}")
    
    # Get all data
    all_data = sheet.get_all_values()
    
    rows_to_fix = []
    
    # Find rows with FALSE text in column N (index 13)
    # Note: Check if it's text "FALSE" by comparing exact value
    for i, row in enumerate(all_data[5:], start=6):  # Skip headers
        if len(row) >= 14:
            end_trade_value = str(row[13]) if len(row) > 13 else ""
            # Check if it's exactly the string "FALSE" or "false"
            # Checkboxes return boolean, text returns string
            if end_trade_value == "FALSE" or end_trade_value == "false":
                rows_to_fix.append(i)
    
    if not rows_to_fix:
        print("✅ No FALSE text found, all checkboxes OK!")
        return
    
    print(f"🔧 Found {len(rows_to_fix)} rows to fix: {rows_to_fix}")
    
    # Fix each row
    for idx, row_num in enumerate(rows_to_fix, 1):
        try:
            # Add checkbox validation
            requests = [{
                "setDataValidation": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": row_num - 1,
                        "endRowIndex": row_num,
                        "startColumnIndex": 13,
                        "endColumnIndex": 14
                    },
                    "rule": {
                        "condition": {"type": "BOOLEAN"},
                        "showCustomUi": True
                    }
                }
            }]
            
            sheet.spreadsheet.batch_update({"requests": requests})
            
            # Clear the cell first, then checkbox will default to unchecked
            sheet.update(f'N{row_num}', [['']], value_input_option='USER_ENTERED')
            
            print(f"  ✅ Fixed row {row_num} ({idx}/{len(rows_to_fix)})")
            
            # Delay every 3 rows to avoid quota
            if idx % 3 == 0 and idx < len(rows_to_fix):
                print(f"  ⏳ Pausing 3s...")
                time.sleep(3)
                
        except Exception as e:
            print(f"  ❌ Error row {row_num}: {e}")
            time.sleep(5)  # Wait longer on error
    
    print(f"\n✅ Fixed {len(rows_to_fix)} rows!")

if __name__ == "__main__":
    fix_checkboxes()
