# Germany-Negative-Price-Analysis

The analysis that this code supported was published in a Linkedin post: https://www.linkedin.com/posts/siddhikulkarni_energytransition-electricitymarkets-renewableenergy-ugcPost-7450152232899158016-45bk?utm_source=share&utm_medium=member_desktop&rcm=ACoAABThqjwBVElJjTIayJAYyJixR9W9glbWtSw

Core research question: The number of negative hours/year in the German day-ahead market are increasing. However, does this translate to progress in the energy transition? Are we  better off today than a few years ago?

For this purpose, 3 metrics are calculated over a period of 3 years and analysed:
- How day-ahead prices have developed season specific. (No. of negative price hours per season)
- How have prices developed in the most stressful times per season. (Price EUR/MWh for the top 5% load hours per season)
- Price spread between the most expensive and the most cheapest hours each season


# Important note on season definition:
The consideration of winter months is not according to a meterological winter. For example Winter 2023 is considered as January, February and December of 2023.

The seasons for a calender year are finally defined as:\
- Winter: January, February, December
- Spring: March, April, May
- Summer: June, July, August
- Autumn: September, October, November

# Data source: Energy-Charts API by Fraunhofer ISE (DE-LU bidding zone)
  https://api.energy-charts.info

# Disclaimer
The Python code in this repository was generated through iterative prompting and debugging with Claude (Anthropic) and ChatGPT (Open AI). This includes the data pipeline, API integrations, metric calculations and data visualisation.

The conceptualization of the research question, analytical framework, interpretation of results and conclusions are entirely my own. AI was used as an assistant and coding tool and not as a thinking tool.

