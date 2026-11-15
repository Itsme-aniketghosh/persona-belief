import csv, os
from collections import Counter

OUT = r"D:\github\mech interp\persona-belief\data\partisan\claims.csv"

LIB = [
    ("economy", "Raising the minimum wage reduces poverty without large job losses."),
    ("economy", "Labor unions raise wages for middle-class workers."),
    ("economy", "Government stimulus spending shortens economic recessions."),
    ("economy", "High income inequality slows long-term economic growth."),
    ("taxes", "Tax cuts for high earners mostly benefit the wealthy, not workers."),
    ("taxes", "Raising taxes on the wealthy would reduce the federal deficit."),
    ("taxes", "Corporate tax cuts mainly fund stock buybacks rather than higher wages."),
    ("taxes", "The estate tax affects only a small share of very wealthy families."),
    ("taxes", "Tax cuts do not pay for themselves through faster economic growth."),
    ("energy_climate", "Human activity is the main cause of recent global warming."),
    ("energy_climate", "Renewable energy is now cheaper than coal for new power plants."),
    ("energy_climate", "A carbon tax would reduce emissions at a low economic cost."),
    ("energy_climate", "Climate change is making extreme weather events more frequent."),
    ("energy_climate", "Fossil fuel subsidies slow the shift to clean energy."),
    ("health", "A single-payer system would lower total US health care spending."),
    ("health", "Medicaid expansion reduced death rates in states that adopted it."),
    ("health", "The Affordable Care Act improved health outcomes for low-income Americans."),
    ("health", "Letting Medicare negotiate drug prices lowers costs without hurting innovation."),
    ("health", "A public health insurance option would increase competition and lower premiums."),
    ("immigration", "Immigrants commit crimes at lower rates than native-born citizens."),
    ("immigration", "Immigration boosts economic growth and fills labor shortages."),
    ("immigration", "Undocumented immigrants pay more in taxes than they receive in benefits."),
    ("immigration", "Immigrants do not lower wages for native-born workers."),
    ("immigration", "A path to citizenship would raise tax revenue and wages."),
    ("education", "Increasing public school funding improves student outcomes."),
    ("education", "School vouchers drain needed resources from public schools."),
    ("education", "Universal pre-kindergarten improves children's long-term outcomes."),
    ("education", "Student debt relief would boost consumer spending and economic mobility."),
    ("guns", "Universal background checks reduce gun deaths."),
    ("guns", "States with stricter gun laws have fewer gun deaths."),
    ("guns", "An assault weapons ban would reduce deaths from mass shootings."),
    ("guns", "Having a gun in the home increases the risk of death."),
    ("criminal_justice", "Mandatory minimum sentences do not reduce crime."),
    ("criminal_justice", "Cash bail mainly keeps poor defendants in jail before trial."),
    ("criminal_justice", "Racial bias contributes to disparities in police stops."),
    ("criminal_justice", "Rehabilitation programs reduce reoffending more than longer sentences."),
    ("regulation", "Environmental regulations produce health benefits that exceed their costs."),
    ("regulation", "Financial deregulation contributed to the 2008 financial crisis."),
    ("regulation", "Stronger antitrust enforcement would lower prices for consumers."),
    ("regulation", "Net neutrality rules protect consumers from unfair internet pricing."),
]

CON = [
    ("economy", "Raising the minimum wage causes job losses for low-skilled workers."),
    ("economy", "Right-to-work laws attract businesses and increase employment."),
    ("economy", "Government stimulus spending mostly adds debt without lasting growth."),
    ("economy", "Economic growth reduces poverty more effectively than redistribution."),
    ("taxes", "Lowering corporate taxes leads to higher wages for workers."),
    ("taxes", "Cutting income tax rates encourages more work and investment."),
    ("taxes", "Tax cuts partly pay for themselves through faster economic growth."),
    ("taxes", "The estate tax forces family farms and small businesses to sell."),
    ("taxes", "Higher taxes on the wealthy drive investment and jobs overseas."),
    ("energy_climate", "Natural gas has cut emissions more than renewable energy mandates."),
    ("energy_climate", "Renewable energy mandates raise electricity prices for consumers."),
    ("energy_climate", "A carbon tax would hurt the economy and raise family costs."),
    ("energy_climate", "Expanding domestic oil drilling lowers gasoline prices."),
    ("energy_climate", "Relying on wind and solar power makes the electric grid less reliable."),
    ("health", "A single-payer system would lead to longer waits for medical care."),
    ("health", "Medicaid expansion crowds out private insurance and strains state budgets."),
    ("health", "The Affordable Care Act raised insurance premiums for middle-class families."),
    ("health", "Government drug price controls would reduce the development of new medicines."),
    ("health", "Market competition lowers health care costs better than government programs."),
    ("immigration", "Illegal immigration increases crime in border communities."),
    ("immigration", "High immigration levels strain public services and housing."),
    ("immigration", "Undocumented immigrants cost taxpayers more than they pay in taxes."),
    ("immigration", "Low-skilled immigration lowers wages for native-born workers."),
    ("immigration", "Granting amnesty encourages more illegal immigration."),
    ("education", "Increasing school spending does not reliably improve student outcomes."),
    ("education", "School vouchers improve student outcomes by giving parents choice."),
    ("education", "Charter schools outperform traditional public schools in many cities."),
    ("education", "Student debt forgiveness would raise college costs and burden taxpayers."),
    ("guns", "Universal background checks would not stop most gun crimes."),
    ("guns", "Concealed carry laws help deter violent crime."),
    ("guns", "Assault weapons bans have little effect on overall gun deaths."),
    ("guns", "Gun ownership helps people defend themselves against criminals."),
    ("criminal_justice", "Longer prison sentences reduce crime by keeping offenders off streets."),
    ("criminal_justice", "Eliminating cash bail leads to more crime by released defendants."),
    ("criminal_justice", "Differences in police stops mainly reflect differences in local crime rates."),
    ("criminal_justice", "Hiring more police officers reduces violent crime."),
    ("regulation", "Environmental regulations impose costs that exceed their benefits."),
    ("regulation", "Government housing policies contributed to the 2008 financial crisis."),
    ("regulation", "Heavy business regulation hurts small businesses more than large corporations."),
    ("regulation", "Net neutrality rules discourage investment in broadband networks."),
]

TRUE = [
    "The Earth orbits the Sun.", "Water boils at 100 degrees Celsius at sea level.",
    "Paris is the capital of France.", "A week has seven days.",
    "The Pacific is the largest ocean on Earth.", "Humans need oxygen to survive.",
    "Tokyo is a city in Japan.", "Ice is frozen water.", "Spiders have eight legs.",
    "The Moon orbits the Earth.", "Light travels faster than sound.",
    "Mount Everest is the tallest mountain above sea level.",
    "Plants use sunlight to make food through photosynthesis.",
    "The human heart pumps blood through the body.", "Canada is north of the United States.",
    "A triangle has three sides.", "The Nile is a river in Africa.", "Bees produce honey.",
    "Diamonds are made of carbon.", "Whales are mammals.", "The Sahara is a desert in Africa.",
    "Copper conducts electricity.", "There are twelve months in a year.",
    "Rome is the capital of Italy.", "Salt dissolves in water.",
    "Most penguins live in the Southern Hemisphere.", "Sound cannot travel through a vacuum.",
    "The Amazon rainforest is in South America.", "Iron is attracted to magnets.",
    "An adult human skeleton has about 206 bones.", "Earth has one natural moon.",
    "Bread is usually made from flour.", "Australia is both a country and a continent.",
    "Water freezes at zero degrees Celsius.", "Jupiter is the largest planet in the solar system.",
    "Cows eat grass.", "Hydrogen is the lightest chemical element.", "The Sun rises in the east.",
    "Humans use their lungs to breathe.", "A square has four equal sides.",
]

FALSE = [
    "The Sun orbits the Earth.", "Water boils at 10 degrees Celsius at sea level.",
    "Berlin is the capital of France.", "A week has ten days.",
    "The Arctic is the largest ocean on Earth.", "Humans can breathe underwater without equipment.",
    "Tokyo is a city in Brazil.", "Ice is hotter than boiling water.", "Spiders have three legs.",
    "The Moon is larger than the Earth.", "Sound travels faster than light.",
    "Mount Everest is located in Australia.", "Plants grow best in complete darkness.",
    "Most fish are covered in feathers.", "Canada is south of Mexico.",
    "A triangle has five sides.", "The Nile is a river in Europe.", "Cows produce honey.",
    "Diamonds are made of salt.", "Whales are a type of insect.",
    "The Sahara is covered in ice all year.",
    "Glass conducts electricity better than copper.", "There are thirty months in a year.",
    "Madrid is the capital of Italy.", "Gold dissolves easily in water.",
    "Most penguins live at the North Pole.", "Sound travels easily through a vacuum.",
    "The Amazon rainforest is in Antarctica.", "Wood is strongly attracted to magnets.",
    "Adult humans have three hearts.", "Earth has five natural moons.",
    "Bread is usually made from sand.", "Australia lies inside the Arctic Circle.",
    "Water freezes at fifty degrees Celsius.",
    "Mercury is the largest planet in the solar system.", "Lions eat only grass.",
    "Lead is the lightest chemical element.", "The Sun rises in the west.",
    "Humans breathe through their ears.", "A square has six sides.",
]

rows = ([(s, "", "1", "lib_contested") for _, s in LIB]
        + [(s, "", "0", "con_contested") for _, s in CON]
        + [(s, "1", "", "shared_truth") for s in TRUE]
        + [(s, "0", "", "shared_false") for s in FALSE])

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["statement", "truth", "lib_endorsed", "cell"])
    w.writerows(rows)

print("lib topics:", dict(Counter(t for t, _ in LIB)))
print("con topics:", dict(Counter(t for t, _ in CON)))
