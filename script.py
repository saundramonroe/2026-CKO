import re

fixes_needed = {
    558: "Change 'merchant_name' to 'merchant'",
    2702: "Change 'return None' to 'pass'",
    304: "Change 'except:' to 'except (requests.exceptions.RequestException, Exception):'",
    311: "Change 'except:' to 'except (requests.exceptions.RequestException, Exception):'",
}

print("Critical fixes needed:")
for line, fix in fixes_needed.items():
    print(f"Line {line}: {fix}")