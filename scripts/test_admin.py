"""Debug: admin_user_ids parsing."""
import os, json

os.environ["ADMIN_USER_IDS"] = "857056289"

# json.loads("857056289") → int 857056289 (not a list, not a string)
value = "857056289"
result = json.loads(value)
print(f"json.loads({value!r}) = {result!r} (type={type(result).__name__})")

# field_validator receives int 857056289
v = result
if isinstance(v, str):
    print("str branch → would parse comma-sep")
elif isinstance(v, list):
    print("list branch → would convert to ints")
else:
    print(f"DEFAULT branch → returns [] ← BUG! (got type={type(v).__name__})")

# The fix: handle int in the validator
print()
print("Now test actual Settings:")
from reviewmind.config import Settings
s = Settings()
print(f"admin_user_ids = {s.admin_user_ids}")
