def part_day(hour):
    if hour >=5 and hour < 11:
        return 'Morning'
    elif hour >= 11 and hour < 17:
        return 'Afternoon'
    elif hour >= 17 and hour < 23:
        return 'Evening'
    else:
        return 'Night'
    
def get_time_periods(hour):
    if hour >= 3 and hour < 7:
        return 'Dawn'
    elif hour >= 7 and hour < 12:
        return 'Morning'
    elif hour >= 12 and hour < 16:
        return 'Afternoon'
    elif hour >= 16 and hour < 22:
        return 'Evening'
    else:
        return 'Night'