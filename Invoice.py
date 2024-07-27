import json
import requests

def invoice_summary (invoice_path):
    

    url = "https://ocr.asprise.com/api/v1/receipt"


    res = requests.post(url,

    data = {

    'api_key': 'TEST',

    'recognizer': 'auto',

    'ref_no': 'oct_python_123'

    },

    files = {

    'file': open(invoice_path, 'rb')

    })
    
    with open( "invoice_response.json",'w') as f:
        json.dump(json.loads(res.text),f)
        
    with open("invoice_response.json",'r') as f:
        data = json.load(f)
    
    bill = data['receipts'][0]
    name = bill['merchant_name'].strip().replace("\n",'')

    summary = f"This is a tax invoice issued to {name} dated {bill['date']}.\n"
    summary += f"The invoice was issued for purchase of a total of {len(bill['items'])} for price {bill['total']}.\n"
    summary += f"Items were bought with the receipt number {bill['receipt_no']} "

    return summary

def invoice_qna (invoice_path):
    url = "https://ocr.asprise.com/api/v1/receipt"


    res = requests.post(url,

    data = {

    'api_key': 'TEST',

    'recognizer': 'auto',

    'ref_no': 'oct_python_123'

    },

    files = {

    'file': open(invoice_path, 'rb')

    })
    
    with open( "invoice_response.json",'w') as f:
        json.dump(json.loads(res.text),f)
        
    with open("invoice_response.json",'r') as f:
        data = json.load(f)
    
    bill = data['receipts'][0]
    
    
    
    return str(bill)
    




