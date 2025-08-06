import datetime


CLASS_DAYS = ['Monday', 'Wednesday', 'Thursday']
DISPLAY_DAYS = CLASS_DAYS
#DISPLAY_DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']


SPECIAL_DATES = [
    (datetime.datetime(2025, 9, 1), 'Labor Day: no classes.'),
    (datetime.datetime(2025, 10, 13), 'Indigenous Peoples\' Day: no classes.'),
    (datetime.datetime(2025, 10, 14), 'Fall Break: no classes.'),
    (datetime.datetime(2025, 10, 28), 'Tanner Conference: no classes.'),
    (datetime.datetime(2025, 11, 26), 'Thanksgiving Break: no classes.'),    
    (datetime.datetime(2025, 11, 27), 'Thanksgiving Break: no classes.'),    
    (datetime.datetime(2025, 11, 28), 'Thanksgiving Break: no classes.'),    
    (datetime.datetime(2025, 12, 11), 'Reading Period Begins.'),    
    (datetime.datetime(2025, 12, 15), 'Final Exam Period Begins.'),    
]


READING_PERIOD_START = datetime.datetime(2025, 12, 11)


def is_date_special(current):
    for d, desc in SPECIAL_DATES:
        if d == current:
            return desc

    return None


def generate_yml_calendar():
    course_start = datetime.datetime(2025, 9, 1)
    course_end = datetime.datetime(2025, 12, 19)

    start = course_start - datetime.timedelta(days=course_start.weekday())
    end = course_end + datetime.timedelta(days=6 - course_end.weekday())

    print('events:')

    current = start
    while current <= end:
        day = current.strftime('%A')

        if day in DISPLAY_DAYS:        
            print('  - month: "{}"'.format(current.strftime('%B')))
            print('    day: "{}"'.format(current.strftime('%d')))
            print('    day-of-week: "{}"'.format(day))

            desc = is_date_special(current)
            if desc is not None:
                print('    special: "{}"'.format(desc))

            if day in CLASS_DAYS:
                print('    topic:')

            print('    due:')
            print('    released:')
            print('    pre-class:')

            if current >= READING_PERIOD_START:
                print('    class-meetings-over: true')
                
            print('')

        current = current + datetime.timedelta(days=1)



def main():    
    #generate_md_calendar()
    #generate_csv_calendar()
    generate_yml_calendar()

    
if __name__ == '__main__':
    main()
