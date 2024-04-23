import datetime


DISPLAY_DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
CLASS_DAYS = ['Monday', 'Wednesday', 'Thursday']


SPECIAL_DATES = [
    (datetime.datetime(2024, 9, 2), 'Labor Day: no classes.'),
    (datetime.datetime(2024, 10, 14), 'Indigenous Peoples’ Day: no classes.'),
    (datetime.datetime(2024, 10, 15), 'Fall Break: no classes.'),
    (datetime.datetime(2024, 10, 29), 'Tanner Conference: no classes.'),
    (datetime.datetime(2024, 11, 27), 'Thanksgiving Break: no classes.'),
    (datetime.datetime(2024, 11, 28), 'Thanksgiving Break: no classes.'),
    (datetime.datetime(2024, 11, 29), 'Thanksgiving Break: no classes.'),
    (datetime.datetime(2024, 12, 11), 'Substitute Day (Lake Day Makeup).'),
    (datetime.datetime(2024, 12, 12), 'Reading Period Begins.'),
    (datetime.datetime(2024, 12, 15), 'Reading Period Ends.'),
]


def is_date_special(current):
    for d, desc in SPECIAL_DATES:
        if d == current:
            return desc

    return None


def generate_md_calendar():
    course_start = datetime.datetime(2024, 9, 2)
    course_end = datetime.datetime(2024, 12, 17)

    start = course_start - datetime.timedelta(days=course_start.weekday())
    end = course_end + datetime.timedelta(days=6 - course_end.weekday())

    print('```{list-table}\n:header-rows: 1\n')
    
    for day in DISPLAY_DAYS:
        prefix = '* -' if day == 'Monday' else '  -'        
        print('{} {}'.format(prefix, day))

    print('')
    
    current = start
    while current <= end:
        day = current.strftime('%A')
        prefix = '* -' if day == 'Monday' else '  -'

        if current < course_start or course_end < current and day in DISPLAY_DAYS:
            print(prefix)            
        elif day in DISPLAY_DAYS:
            print('{} <p class="text-muted">{}</p>'.format(prefix, current.strftime('%d %B')))

            desc = is_date_special(current)
            if desc is not None:
                print('    <strong>{}</strong>'.format(desc))

        if day == 'Sunday':
            print('')
            
        current = current + datetime.timedelta(days=1)

    print('```')


def generate_csv_calendar():
    course_start = datetime.datetime(2024, 9, 2)
    course_end = datetime.datetime(2024, 12, 17)

    start = course_start - datetime.timedelta(days=course_start.weekday())
    end = course_end + datetime.timedelta(days=6 - course_end.weekday())

    print(', '.join([day for day in DISPLAY_DAYS]) + ', ')

    row_date = ''
    row_content = ''
    current = start
    while current <= end:
        day = current.strftime('%A')

        if current < course_start or course_end < current and day in DISPLAY_DAYS:
            row_date += ', '
        elif day in DISPLAY_DAYS:
            row_date += current.strftime('%d %B') + ', '

        if day in DISPLAY_DAYS:
            desc = is_date_special(current)
            if desc is None:
                row_content += ','
            else:
                row_content += desc + ', '
            
        if day == 'Sunday':
            print(row_date)
            print(row_content)
            row_date = ''
            row_content = ''
            
        current = current + datetime.timedelta(days=1)


def generate_yml_calendar():
    course_start = datetime.datetime(2024, 9, 2)
    course_end = datetime.datetime(2024, 12, 17)

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
            print('')

        current = current + datetime.timedelta(days=1)



def main():    
    #generate_md_calendar()
    #generate_csv_calendar()
    generate_yml_calendar()

    
if __name__ == '__main__':
    main()
