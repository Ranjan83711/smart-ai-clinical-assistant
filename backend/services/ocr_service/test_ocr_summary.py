from .summarize_report import interpret_medical_report

image_path = "services/ocr_service/image.png"

result = interpret_medical_report(image_path)

print("\n=========== RAW OCR TEXT ===========")
print(result["raw_ocr"])

print("\n=========== FINAL INTERPRETATION ===========")
print(result["explanation"])