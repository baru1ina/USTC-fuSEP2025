import datetime as dt
import pytz
from zoneinfo import ZoneInfo

# # Все доступные часовые пояса
# all_timezones = pytz.all_timezones
#
# with open("timezones.txt", "w", encoding="utf-8") as f:
#     f.write("\n".join(all_timezones))

with open("logout.txt", 'a') as file:
    timezone = pytz.timezone("Asia/Shanghai")
    current_time_CN = dt.datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S %Z")
    current_time_RU = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
    author = "Sofia"
    file.write(f"[LOG] Modified time CN: {current_time_CN}\n"
               f"[LOG] Modified time RU: {current_time_RU}\n"
               f"[LOG] Editor: {author}\n\n")

