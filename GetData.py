import csv
import requests

#FBS ONLY
GAMES_URL = 'http://masseyratings.com/scores.php?s=300937&sub=11604&all=1&mode=3&format=1'
TEAMS_URL = 'http://masseyratings.com/scores.php?s=300937&sub=11604&all=1&mode=3&format=2'

# GAMES_URL = 'https://www.masseyratings.com/scores.php?s=300937&sub=11590&all=1&mode=3&format=1'
# TEAMS_URL = 'https://www.masseyratings.com/scores.php?s=300937&sub=11590&all=1&mode=3&format=2'


def returnTeams():
    #create a session
    with requests.Session() as s:
        download = s.get(GAMES_URL)

        decoded_content = download.content.decode('utf-8')

        #seperate out the csv
        cr = csv.reader(decoded_content.splitlines(), delimiter=',')
        my_list = list(cr)
        formatted_games = []
        for row in my_list:
            row = [int(x.strip(' ')) for x in row]
            formatted_games.append(row)

        download = s.get(TEAMS_URL)
        decoded_content = download.content.decode('utf-8')
        cr = csv.reader(decoded_content.splitlines(), delimiter=',')
        my_list = list(cr)
        formatted_teams = {}

        for row in my_list:
            row = [int(row[0].strip(' ')), row[1].strip(' ')]
            formatted_teams[row[0]] = row[1]
        return formatted_games,formatted_teams


if __name__ == "__main__":
    print (returnTeams())