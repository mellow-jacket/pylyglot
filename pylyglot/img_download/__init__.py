'''
 Separating scrape functions here for now.
 Each website/comic having its own file for now
'''


from .comic_eng import scrape as scrape_md
from .comic_kor import scrape as scrape
from .novel_kor import scrape as scrape_novel