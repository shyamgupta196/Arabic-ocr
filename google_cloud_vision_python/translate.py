import os

# Imports the Google Cloud Translation library
from google.cloud import translate

# Initialize Translation client
def translate_text(
    text: str = "YOUR_TEXT_TO_TRANSLATE", project_id: str = "gcp-kubernetes-ml-app"
) -> translate.TranslationServiceClient:
    """Translating Text."""

    client = translate.TranslationServiceClient()

    location = "global"

    parent = f"projects/{project_id}/locations/{location}"

    # Translate text from English to French
    # Detail on supported types can be found here:
    # https://cloud.google.com/translate/docs/supported-formats
    response = client.translate_text(
        request={
            "parent": parent,
            "contents": [text],
            "mime_type": "text/plain",  # mime types: text/plain, text/html
            "source_language_code": "ar",
            "target_language_code": "en",
        }
    )

    # Display the translation for each input text provided
    for translation in response.translations:
        print(f"Translated text: {translation.translated_text}")

    return response


if __name__ == "__main__":
    file = open("outputs\SHUKRAN.txt", "r", encoding="utf-8")
    translate_text(
        """

texts:G
Tax Invoice
Invoice Date:
Due Date:
Delivery Date:
Source:
Reference:
000011088
رقم الإشاره :
DESCRIPTION
الوصف
كرنشي كيك B2B |
07/04/2023
07/04/2023
07/04/2023
502090
[B2B Section ganache ] [Big Cake] Crunchy Chocolate
الإجمالي الفرعي / Subtotal |
| Taxes 15%
المجموع / Total
Payment Reference: INV/2023/02064
Payment terms: Immediate Payment
203
INV/2023/02064
تاريخ الفاتورة :
تاريخ الاستحقاق :
تاريخ التوصيل :
المصدر :
QUANTITY
الكميه
1.000 Units
132.00 SR |
19.80 SR |
151.80 SR
UNIT
PRICE
مؤسسة انس غالب حمزه خاشقجي التجاريه
RATIONS
سعر
الضرائب الوحدة
شركة الامجاد للتجارة والمقاولات / BARNS cafe
203 MALL مکه BARN
حده
TAXES AMOUNT
151.80 Sales Tax | 132.00 SR
15%
bars
203,
GANACHE
مبلغ
فاتورة ضريبية
Rawdah Street
Jeddah Saudi Arabia
qasim@ganache-sa.com http://ganache-sa.com 300114827900003
Page: 1 / 1
VAT
AMOUNT
TOTAL
PRICE
قمه
السعر
الاحمالي الصربية
19.80 151.80 SR
رقم إشارة الدفعة : 2023/02064/INV
شروط السداد: سداد فوري
    """
    )
