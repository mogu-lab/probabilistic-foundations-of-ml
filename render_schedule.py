import yaml
import dominate
import dominate.tags as tags
from scripts.generate_base_schedule import DISPLAY_DAYS, CLASS_DAYS 


def main():
    with open('events.yml', 'r') as f:
        events = yaml.safe_load(f)['events']

    doc =tags.table(cls='table course_calendar')

    with doc.add(tags.tbody()):
        with tags.thead(cls='col_headers'):
            header = tags.tr()
            for day in DISPLAY_DAYS:
                header += tags.th(day, scope='col')

        row = None
        cur_month = ''
        for event in events:
            if event['day-of-week'] == DISPLAY_DAYS[0]:
                row = tags.tr(cls='light_row')

            special = event.get('special', None)
            pre_class = event.get('pre-class', None)
            topic = event.get('topic', None)
            due = event.get('due', None)
            released = event.get('released', None)

            td_tags = 'normalday' if special is None else 'holiday'
            if event['month'] != cur_month:
                cur_month = event['month']
                td_tags += ' new_month'
                
            with row.add(tags.td(cls=td_tags)):                    
                tags.span(event['day'], cls='date_label_day')              
                tags.span(event['month'], cls='date_label_month')

                with tags.ul(cls='day_agenda'):
                    if special is not None:
                        tags.li(tags.span(special, cls='day_note'))

                    if pre_class is not None:
                        tags.li('Pre-class work: {}'.format(pre_class))
                        
                    if topic is not None:
                        tags.li('Topic: {}'.format(topic))

                    if due is not None:
                        tags.li('Due: {}'.format(due))
                        
                    if released is not None:
                        tags.li('Released: {}'.format(released))
                    
                    
    print('# Schedule')
    print('')
    print(doc)
    

if __name__ == '__main__':
    main()

    
