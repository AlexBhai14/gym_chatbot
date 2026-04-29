
from __future__ import annotations
import json, os, re
from retriever import get_retriever
from generator import get_generator

TOP_K = 3
OUT_OF_DOMAIN_THRESHOLD = 0.30

OUT_OF_DOMAIN_ANSWER = (
    "I am a Gym & Fitness Assistant and can only answer questions related to "
    "workouts, diet, supplements, gym membership, fitness goals, and health. "
    "Please ask me a fitness-related question and I will be happy to help! 💪"
)

NON_FITNESS_KEYWORDS = {
    # Date & Time
    "today date","current date","what date","what day","what time",
    "time now","date today","day today","year today","current time",
    "what year","what month","aaj ki date","aaj ka din","aaj ka time",
    "abhi ka time","kitne baje","date kya hai","din kya hai","time kya hai",
    # CS / Tech
    "lifo","fifo","data structure","linked list","algorithm","bubble sort",
    "merge sort","recursion","python code","python program","javascript",
    "html","css","react","nodejs","machine learning","deep learning",
    "neural network","artificial intelligence","chatgpt","openai",
    "sql","docker","github","operating system","linux","ubuntu",
    # Finance
    "stock market","share market","nifty","sensex","mutual fund",
    "bitcoin","crypto","ethereum","blockchain","trading","forex",
    "bank interest","home loan","income tax","gst","itr",
    # Weather
    "weather today","aaj ka mausam","barish kab","temperature today",
    "mausam kaisa hai","weather report","humidity",
    # Cricket / Sports news
    "cricket score","ipl score","test match score","odi match",
    "football score","isl score","fifa","sports news",
    # Entertainment
    "bollywood movie","web series","netflix","amazon prime",
    "song lyrics","bollywood actor","new movie",
    # Politics / News
    "news today","breaking news","aaj ki khabar","election result",
    "election date","bjp","congress party","prime minister kaun hai",
    # General knowledge
    "capital of","president of","history of","population of",
    "who invented","who discovered","world war",
    # Religion / Astrology
    "rashifal","horoscope","kundli","astrology","zodiac sign",
    "pooja vidhi","vrat kab","festival date",
    # Education / Govt
    "neet exam","jee exam","upsc","board exam","sarkari naukri",
    "aadhar card","pan card","passport","driving license",
    "ration card","voter id","train ticket","flight ticket",
    # Relationships
    "love problem","girlfriend kaise","boyfriend ne","relationship tips",
    # Cooking non-fitness
    "biryani recipe","butter chicken","cake kaise","pizza recipe",
    "halwa recipe","gulab jamun","kheer recipe",
    # Math / Science
    "square root","trigonometry","photosynthesis","gravity kya hai",
    "speed of light","e equals mc",

     # ── DATE & TIME (variants not in original) ──────────────────────
    "aaj kaun sa din hai","abhi kya time hai","kal kaun sa din hai",
    "parso kya din hai","is hafte kaun sa din","mahine ki date",
    "aaj ki tarikh","tarikh batao","din aur tarikh","what is today",
    "tell me the date","what time is it now","current date and time",
    "today is which day","which month is running","which year is this",
    "aaj sunday hai ya nahi","is it monday today","what day of week",
    "date and time please","abhi kaun sa mahina hai","kon sa saal hai",
    "aaj ka schedule","calendar today","monthly calendar",
    "aaj kitni tarikh hai","aaj ki date batao","what is the time right now",
 
    # ── COMPUTER SCIENCE (new variants) ──────────────────────────────
    "what is oops","object oriented programming","what is class in java",
    "what is inheritance","what is polymorphism","what is abstraction",
    "what is encapsulation","what is array","what is pointer in c",
    "stack vs queue","binary search","time complexity","space complexity",
    "big o notation","what is compiler","what is interpreter",
    "what is api rest","what is http","what is tcp ip","what is dns",
    "what is url","what is ip address","what is subnet mask",
    "what is firewall","what is vpn","what is proxy",
    "what is cloud","aws kya hai","google cloud","microsoft azure",
    "what is devops","what is agile","what is scrum","what is git",
    "git commands","what is branch in git","how to commit in git",
    "what is numpy","what is pandas library","what is tensorflow",
    "what is pytorch","what is keras","what is scikit learn",
    "what is flask","what is django","what is fastapi",
    "what is mongodb","what is postgresql","what is redis",
    "what is microservices","what is kubernetes","what is jenkins",
    "what is selenium","what is junit","what is unit testing",
    "coding interview questions","dsa questions","leetcode problem",
    "hackerrank problem","system design interview","low level design",
    "what is recursion in python","linked list in c++","tree traversal",
    "graph algorithm","dijkstra algorithm","dynamic programming",
    "what is blockchain technology","smart contract kya hai",
    "what is nft token","web3 kya hai","decentralized app",
    "what is cybersecurity","ethical hacking kaise sikhe",
    "kali linux kya hai","penetration testing","sql injection",
    "what is excel formula","vlookup kaise kare","pivot table kya hai",
    "powerpoint presentation kaise banaye","ms word shortcut keys",
    "google sheets formula","what is chatgpt 4","gpt kya hai",
    "midjourney kya hai","stable diffusion","ai image generator",
    "what is prompt engineering","llm kya hai","rag kya hai in ai",
    "what is fine tuning model","what is langchain",
    "mobile app development","flutter kya hai","react native",
    "android development","ios development","swift programming",
    "kotlin kya hai","dart programming","xamarin kya hai",
    "what is figma","adobe xd kya hai","ui ux design kaise sikhe",
    "website kaise banaye","wordpress kaise use kare","wix vs wordpress",
    "shopify store kaise banaye","seo kaise kare","digital marketing",
    "google ads kaise chalaye","facebook ads","instagram marketing",
    "affiliate marketing kya hai","dropshipping kya hai",
    "freelancing kaise kare","upwork profile kaise banaye",
    "fiverr se paisa kaise kamaye","youtube channel kaise banaye",
 
    # ── FINANCE & MONEY (new variants) ───────────────────────────────
    "share kaise kharide zerodha mein","demat account kaise khole",
    "zerodha kaise use kare","groww app kaise use kare",
    "upstox kaise use kare","angel broking","5paisa kya hai",
    "nifty 50 kya hai","sensex aaj ka","bse nse difference",
    "ipo mein invest kaise kare","grey market premium","ipo allotment",
    "mutual fund sip kaise shuru kare","elss fund kya hai",
    "index fund vs active fund","nifty bees kya hai",
    "gold etf kya hai","sovereign gold bond","real estate investment",
    "reits kya hai","reit in india","infrastructure investment trust",
    "p2p lending kya hai","fixed deposit interest rate",
    "recurring deposit kya hai","post office scheme",
    "national savings certificate","public provident fund ppf",
    "sukanya samridhi yojana","atal pension yojana","nps kya hai",
    "epf kaise check kare","pf balance check","epfo login",
    "gratuity calculation","salary structure in india",
    "in hand salary calculator","ctc vs take home","tds kya hai",
    "form 16 kya hota hai","80c deduction kya hai",
    "income tax slab 2024","old vs new tax regime",
    "capital gains tax kya hai","long term capital gain",
    "how to file itr online","itr 1 vs itr 2","tax refund kaise milega",
    "gst registration kaise kare","gst return filing",
    "udyam registration kya hai","msme loan kaise milega",
    "mudra loan kaise apply kare","startup india scheme",
    "term insurance kya hai","life insurance vs term insurance",
    "health insurance kaise choose kare","mediclaim policy",
    "motor insurance claim kaise kare","endowment policy kya hai",
    "ulip kya hai","lic policy surrender kaise kare",
    "credit score kaise badhaye","cibil score check kaise kare",
    "credit card bill kaise bhare","credit card limit badhana",
    "emi calculator","personal loan interest rate",
    "home loan eligibility calculator","car loan kaise milega",
    "education loan kaise apply kare","loan against property",
    "bitcoin kaise kharide india mein","crypto exchange india",
    "wazirx kya hai","coindcx kya hai","crypto tax in india",
    "ethereum kaise kharide","dogecoin kya hai","shiba inu coin",
    "defi kya hai","yield farming","liquidity pool",
    "forex trading kaise kare","dollar vs rupee rate today",
    "gold price today india","silver price today","commodity market",
    "mcx trading kya hai","futures and options trading",
    "options chain kaise padhein","call put option kya hai",
    "intraday trading kaise kare","swing trading kya hai",
    "warren buffett investment tips","benjamin graham value investing",
 
    # ── WEATHER (new variants) ────────────────────────────────────────
    "aaj barish hogi kya","kal ka mausam kaisa rahega",
    "is week mein rain hogi","monsoon kab aayega",
    "cyclone news today","earthquake news today",
    "flood situation india","drought kahan hai",
    "heat wave alert today","cold wave warning",
    "temperature in delhi today","mumbai mausam aaj",
    "bangalore weather forecast","chennai temperature",
    "kolkata weather today","hyderabad mausam",
    "shimla snowfall today","manali mein barish",
    "air quality index today","aqi delhi today",
    "pollution level today","smog alert","uv index today",
    "sunrise time today","sunset time today","golden hour time",
 
    # ── CRICKET & SPORTS NEWS (new variants) ─────────────────────────
    "ipl 2024 schedule","ipl today match","ipl points table",
    "ipl auction 2024","csk vs mi score","rcb match today",
    "virat kohli score today","rohit sharma news","ms dhoni news",
    "india vs australia test","india vs england score",
    "world cup 2024 cricket","t20 world cup schedule",
    "champions trophy 2025","asia cup schedule",
    "ranji trophy news","vijay hazare trophy",
    "bbl cricket score","county cricket news",
    "icc ranking today","test match playing 11",
    "indian premier league winner","purple cap orange cap",
    "football world cup 2026","premier league score today",
    "la liga results","champions league score",
    "bundesliga news","serie a results","ligue 1 score",
    "manchester united news","liverpool score today",
    "real madrid news","barcelona match score",
    "messi news today","ronaldo news today","neymar news",
    "fifa ranking today","euro cup 2024",
    "nba score today","lakers vs warriors","lebron james news",
    "stephen curry stats","nba playoffs 2024",
    "formula 1 race today","f1 standings 2024",
    "lewis hamilton news","max verstappen news","ferrari f1",
    "badminton world championship","pv sindhu news",
    "kidambi srikanth score","saina nehwal news",
    "chess world championship","magnus carlsen news",
    "viswanathan anand news","r praggnanandhaa news",
    "boxing match today","ufc fight results",
    "wrestling wwe results","kabaddi pro score",
    "hockey world cup india","kho kho championship",
    "olympics 2024 medal tally","paris olympics results",
    "asian games india medals","commonwealth games",
    "indian army sports news","khelo india games",
 
    # ── ENTERTAINMENT (new variants) ──────────────────────────────────
    "salman khan new movie","shah rukh khan film",
    "aamir khan movie 2024","akshay kumar upcoming film",
    "ranveer singh news","deepika padukone movie",
    "priyanka chopra news","katrina kaif film",
    "alia bhatt upcoming movie","ranbir kapoor news",
    "hrithik roshan new film","tiger shroff upcoming movie",
    "ayushmann khurrana news","kartik aaryan film",
    "taapsee pannu movie","kangana ranaut news",
    "south indian movies 2024","prabhas new film",
    "allu arjun movie","ram charan news","jr ntr film",
    "vijay thalapathy movie","ajith kumar film",
    "thalapathy vijay news","dhanush new movie",
    "tamil movies 2024","telugu box office",
    "kannada movie news","malayalam films 2024",
    "box office collection today","highest grossing bollywood film",
    "oscar nominations 2024","grammy awards 2024",
    "filmfare awards 2024","zee cine awards",
    "bigg boss season 17","bigg boss winner",
    "kaun banega crorepati","dance plus season",
    "india got talent","indian idol winner",
    "ott release this week","netflix india new shows",
    "amazon prime new series","hotstar web series",
    "zee5 new show","sony liv series","jio cinema content",
    "squid game season 2","stranger things news",
    "game of thrones prequel","the last of us news",
    "marvel new movie","avengers next film","dc universe news",
    "spider-man new film","batman news","superman movie",
    "taylor swift new song","ed sheeran album","arijit singh song",
    "atif aslam new song","badshah new rap","diljit dosanjh concert",
    "ap dhillon song","divine rapper news","raftaar new song",
    "punjabi song new 2024","haryanvi song latest",
    "bhojpuri song 2024","rajasthani folk music",
    "spotify wrapped","youtube music charts","gaana trending songs",
    "tik tok ban india","instagram new feature","youtube shorts",
    "reels viral today","meme of the day","trending on twitter",
 
    # ── POLITICS & GOVERNMENT (new variants) ──────────────────────────
    "narendra modi news","rahul gandhi speech","amit shah news",
    "arvind kejriwal news","mamata banerjee","yogi adityanath news",
    "cm of maharashtra","cm of rajasthan","cm of up",
    "lok sabha election 2024","rajya sabha session",
    "parliament news today","budget 2024 india",
    "supreme court verdict today","high court decision",
    "cbi raid news","ed investigation news",
    "income tax raid news","enforcement directorate",
    "niti aayog news","planning commission",
    "rbi news today","rbi rate hike","repo rate change",
    "inflation rate india","gdp growth rate",
    "unemployment rate india","poverty rate news",
    "demonetization effect","gst council meeting",
    "petrol diesel price today","lpg cylinder price",
    "cng price today","electricity bill hike",
    "water supply news","metro rail news",
    "bullet train india update","vande bharat express",
    "atal setu bridge","ram mandir news",
    "manipur situation news","kashmir news today",
    "border dispute china india","pak india relations",
    "ukraine russia war update","israel hamas war",
    "america india relations","modi biden meeting",
    "g20 summit news","brics news","quad meeting",
    "united nations india","who india","world bank india",
    "imf india forecast","asian development bank",
 
    # ── GENERAL KNOWLEDGE (new variants) ──────────────────────────────
    "who is the richest person in world","elon musk net worth",
    "mukesh ambani wealth","gautam adani net worth",
    "forbes list 2024 india","hurun rich list",
    "largest country in world","smallest country",
    "deepest ocean in world","highest mountain after everest",
    "longest river in india","largest desert in india",
    "national bird of india","national animal","national flower",
    "national fruit of india","national game of india",
    "constitution of india article 370","fundamental rights india",
    "directive principles","preamble of india",
    "who wrote constitution of india","drafting committee chairman",
    "first prime minister of india","first president of india",
    "first woman prime minister","indira gandhi information",
    "bhagat singh history","subhash chandra bose",
    "mahatma gandhi biography","jawaharlal nehru",
    "ambedkar contribution","sardar vallabhbhai patel",
    "1857 revolt history","independence day 1947",
    "partition of india history","mughal empire history",
    "akbar the great history","aurangzeb history",
    "maratha empire history","shivaji maharaj",
    "chhatrapati sambhaji","peshwa history",
    "chandragupta maurya","ashoka the great",
    "gupta empire history","harappan civilization",
    "indus valley civilization","vedic period history",
    "ramayana author","mahabharata author",
    "who wrote bhagavad gita","who wrote arthashastra",
    "periodic table elements","chemical formula water",
    "atomic number of gold","molecular weight of co2",
    "newton's laws of motion","einstein theory of relativity",
    "darwin theory of evolution","dna structure discovered by",
    "who invented electricity","who invented telephone",
    "who invented computer","who invented internet",
    "who invented airplane","wright brothers facts",
    "who invented penicillin","who discovered gravity",
    "who discovered america","christopher columbus",
    "world war 1 start date","world war 2 end date",
    "hiroshima nagasaki bomb","cold war history",
    "french revolution year","russian revolution",
    "chinese revolution history","partition of korea",
    "vietnam war history","gulf war history",
    "moon landing year","neil armstrong biography",
    "space x kya hai","isro upcoming mission",
    "chandrayaan 3 update","gaganyaan mission india",
    "mars mission nasa","james webb telescope news",
    "black hole discovery","gravitational waves",
 
    # ── RELIGION & SPIRITUALITY (new variants) ────────────────────────
    "ram navami 2024 date","navratri 2024 kab hai",
    "diwali 2024 date","holi 2024 kab hai",
    "eid ul fitr 2024","bakrid 2024 date",
    "christmas 2024","good friday 2024",
    "guru nanak jayanti","buddha purnima 2024",
    "mahavir jayanti","parsi new year",
    "shivratri 2024 date","janmashtami 2024",
    "ganesh chaturthi 2024","durga puja 2024",
    "chhath puja 2024","karwa chauth 2024",
    "makar sankranti date","lohri 2024 date",
    "pongal 2024 date","onam 2024 date",
    "bihu festival date","ugadi 2024",
    "gudi padwa 2024","baisakhi 2024 date",
    "kumbh mela 2025","char dham yatra",
    "vaishno devi darshan","tirupati balaji darshan",
    "shirdi sai baba","rishikesh ashram",
    "rameshwaram temple","jagannath puri yatra",
    "shiv chalisa","hanuman chalisa lyrics",
    "gayatri mantra meaning","mahamrityunjay mantra",
    "ganesh aarti lyrics","lakshmi aarti words",
    "vishnu sahasranam","sunderkand path",
    "geeta adhyay 2","bhagwat puran katha",
    "ramcharitmanas chaupai","quran ayat meaning",
    "hadith in hindi","namaz kaise ada kare",
    "roza rakhne ke niyam","zakat kya hai",
    "bible verses hindi mein","church near me timing",
    "guru granth sahib path","ardas kaise kare",
    "gurdwara near me","amritsar golden temple timing",
    "jain festival paryushan","mahavir swami story",
    "buddhist meditation technique","vipassana meditation",
    "feng shui for home","vastu shastra bedroom",
    "numerology name calculator","tarot card reading online",
    "palmistry hand reading","birth chart astrology",
    "kundli milan kaise kare","lal kitab remedies",
    "shani sade sati kya hai","rahu ketu effects",
    "manglik dosh kya hai","gemstone for luck",
    "rudraksha benefits","yantra kya hota hai",
 
    # ── EDUCATION & CAREER (new variants) ────────────────────────────
    "neet 2024 cutoff","jee main 2024 result",
    "jee advanced 2024","iit admission process",
    "nit admission criteria","aiims mbbs seats",
    "clat 2024 preparation","law entrance exam",
    "cat 2024 exam date","mba admission process",
    "iim fees structure","iim bangalore admission",
    "gate 2024 exam","gate civil engineering",
    "upsc syllabus 2024","upsc prelims date",
    "ias preparation books","ips officer salary",
    "ifs exam kya hai","iras officer information",
    "ssc cgl 2024","ssc chsl exam date",
    "bank po exam 2024","ibps clerk notification",
    "rbi grade b exam","nabard exam 2024",
    "rrb ntpc recruitment","railway group d 2024",
    "army agniveer bharti","navy recruitment 2024",
    "air force x y group","crpf constable vacancy",
    "cisf recruitment 2024","ssb interview tips",
    "state psc exam list","mpsc exam date",
    "rpsc exam 2024","bpsc 69th notification",
    "tet exam 2024","ctet exam date",
    "net exam june 2024","ugc net syllabus",
    "phd admission 2024","research fellowship",
    "scholarship for engineering","merit scholarship india",
    "study abroad process","gre exam syllabus",
    "gmat preparation","ielts vs toefl",
    "sat exam india","act exam syllabus",
    "us university admission","uk university admission",
    "canada pr process","australia immigration",
    "germany study visa","france scholarship",
    "10th ke baad kya kare","12th commerce ke baad",
    "science ke baad career","arts mein career options",
    "btech vs bsc which better","engineering vs medical",
    "ca exam date","cs exam 2024","cma exam",
    "data science career","artificial intelligence jobs",
    "cybersecurity career india","ethical hacker salary",
    "product manager kaise bane","ux designer salary india",
    "content writing career","social media manager salary",
    "graphic designer tools","video editing software",
    "resume kaise banaye","cover letter format",
    "linkedin profile tips","job interview tips",
    "hr interview questions","salary negotiation tips",
    "how to get first job","internship kaise milega",
    "campus placement tips","off campus placement",
 
    # ── GOVERNMENT SCHEMES & DOCUMENTS (new variants) ─────────────────
    "pm kisan samman nidhi","pm awas yojana registration",
    "pm ujjwala yojana","jan dhan account opening",
    "ayushman bharat card","e shram card registration",
    "ration card new apply","ration card update",
    "birth certificate online apply","death certificate apply",
    "income certificate kaise banaye","domicile certificate",
    "caste certificate sc st obc","obc certificate validity",
    "marriage certificate apply","divorce process india",
    "name change after marriage","passport renewal process",
    "passport tatkal service","e passport kya hai",
    "visa application process","us visa appointment",
    "schengen visa india","australia visa process",
    "canada visitor visa","dubai visa requirements",
    "aadhar update mobile number","aadhar pan link",
    "pan card correction","pan card lost replacement",
    "voter id download online","voter id name correction",
    "driving license renewal","learner license apply",
    "rc book transfer process","fitness certificate vehicle",
    "insurance renewal online","third party insurance",
    "vehicle registration check","fastag recharge",
    "challan kaise bhare","traffic challan check",
    "property registration process","registry kaise hoti hai",
    "stamp duty calculation","khata certificate",
    "rera registration","building plan approval",
    "electricity meter connection","solar panel subsidy",
    "water connection apply","sewage connection",
    "fire noc application","trade license apply",
    "shop establishment certificate","fssai license",
    "drug license apply","pollution certificate vehicle",
    "msme registration online","startup registration process",
    "trademark registration india","copyright registration",
    "patent filing india","design registration",
 
    # ── TRAVEL & TOURISM (new variants) ──────────────────────────────
    "goa tour package price","manali trip budget",
    "kerala tour package","rajasthan tour plan",
    "leh ladakh bike trip","spiti valley tour",
    "andaman nicobar package","kashmir tour 2024",
    "ooty kodaikanal trip","coorg tour package",
    "rishikesh adventure sports","haridwar ghat darshan",
    "varanasi ghat kashi","ayodhya ram mandir visit",
    "mathura vrindavan tour","ujjain mahakaleshwar",
    "shirdi package from mumbai","tirupati darshan ticket",
    "golden temple amritsar","vaishno devi helicopter",
    "char dham yatra cost","kedarnath trek route",
    "badrinath opening date","gangotri yamunotri",
    "puri jagannath rath yatra","konark sun temple",
    "hampi karnataka tour","mysore palace visit",
    "mahabaleshwar strawberry season","lonavala khandala",
    "mumbai tourist places","delhi sightseeing",
    "agra taj mahal ticket","red fort ticket price",
    "qutub minar timing","india gate area",
    "jaipur pink city tour","udaipur lake palace",
    "jodhpur blue city","jaisalmer desert safari",
    "pushkar camel fair","rann of kutch festival",
    "sundarban tour west bengal","kaziranga national park",
    "jim corbett tiger reserve","ranthambore safari",
    "kanha national park","pench safari booking",
    "bandipur wildlife safari","nagarhole park",
    "dudhwa national park","sariska tiger reserve",
    "bhitarkanika mangroves","chilika lake birds",
    "international travel tips","duty free allowance india",
    "travel insurance kya hai","forex card vs cash",
    "cheapest flight booking site","irctc tatkal booking",
    "bus booking app india","ola uber outstation",
    "hotel booking discount tips","airbnb india",
    "hostel booking india backpacker","solo travel india tips",
    "travel vlog youtube india","budget travel india",
 
    # ── HEALTH (NON-FITNESS) ──────────────────────────────────────────
    "covid symptoms 2024","corona new variant",
    "dengue fever treatment","malaria prevention",
    "typhoid symptoms hindi","chikungunya treatment",
    "swine flu symptoms","bird flu india",
    "diabetes type 2 diet","sugar ki bimari",
    "blood pressure medicine","hypertension treatment",
    "thyroid symptoms women","hypothyroid diet",
    "thyroid medication","tsh level normal range",
    "pcod vs pcos difference","pcos treatment medicine",
    "kidney stone treatment","gall bladder stone",
    "appendix operation","hernia operation cost",
    "piles treatment india","fissure treatment",
    "acidity treatment hindi","gerd symptoms",
    "ibs irritable bowel","crohn's disease",
    "ulcerative colitis diet","liver disease treatment",
    "fatty liver treatment","hepatitis b treatment",
    "jaundice symptoms hindi","yellow fever",
    "dengue platelet count","white blood cell count",
    "hemoglobin low treatment","anemia symptoms hindi",
    "iron deficiency diet","vitamin b12 deficiency",
    "calcium deficiency symptoms","vitamin d deficiency",
    "arthritis treatment hindi","joint pain medicine",
    "back pain treatment","slip disc operation",
    "migraine treatment","headache tablet name",
    "anxiety disorder treatment","depression medicine india",
    "insomnia treatment natural","sleep disorder doctor",
    "skin allergy treatment","eczema treatment hindi",
    "psoriasis treatment","vitiligo safed daag",
    "acne treatment hindi","pimple cream india",
    "hair fall treatment medicine","alopecia treatment",
    "dandruff treatment shampoo","grey hair causes",
    "eye flu treatment","conjunctivitis symptoms",
    "contact lens vs glasses","lasik surgery cost india",
    "dental crown cost india","root canal treatment",
    "braces cost in india","teeth whitening india",
    "hearing loss treatment","tinnitus cure",
    "cancer symptoms early","breast cancer awareness",
    "cervical cancer vaccine","prostate cancer symptoms",
    "lung cancer symptoms","skin cancer types",
    "chemotherapy side effects","radiation therapy",
    "ayurvedic medicine for diabetes","homeopathy for thyroid",
    "naturopathy treatment","unani medicine",
    "homeopathy doctor near me","ayurveda panchkarma",
 
    # ── FOOD & COOKING (NON-FITNESS) ──────────────────────────────────
    "chicken biryani recipe","mutton biryani kaise banaye",
    "egg biryani recipe","veg biryani recipe",
    "butter chicken gravy","chicken tikka masala",
    "palak paneer recipe","shahi paneer recipe",
    "dal makhani recipe","chana masala recipe",
    "rajma recipe","matar paneer recipe",
    "aloo gobi sabzi","baingan bharta recipe",
    "chole bhature recipe","puri bhaji recipe",
    "paratha recipe aloo","methi paratha recipe",
    "roti kaise banaye soft","naan recipe tandoor",
    "samosa recipe crispy","kachori recipe",
    "pakora recipe crispy","bhajiya recipe",
    "idli recipe soft","dosa batter recipe",
    "sambhar recipe","coconut chutney recipe",
    "vada pav recipe","pav bhaji recipe",
    "misal pav recipe","poha recipe easy",
    "upma recipe","sheera recipe",
    "khichdi recipe","halwa sooji recipe",
    "kheer rice recipe","gajar ka halwa",
    "gulab jamun recipe","jalebi recipe",
    "ladoo recipe besan","barfi recipe kaju",
    "rasgulla recipe","sandesh recipe bengali",
    "raita recipe boondi","lassi recipe",
    "chaas buttermilk recipe","thandai recipe",
    "masala chai recipe","filter coffee",
    "cold coffee recipe","mango shake recipe",
    "lemon rice recipe","tamarind rice",
    "fried rice recipe","hakka noodles recipe",
    "chowmein recipe","pasta recipe indian style",
    "pizza dough recipe","sandwich recipe",
    "cake recipe without oven","eggless cake recipe",
    "brownie recipe easy","cookies recipe",
    "ice cream recipe","kulfi recipe easy",
    "mango pickle recipe","nimbu achar",
    "chutney recipe mint","imli chutney recipe",
 
    # ── LIFESTYLE & HOME (new topics) ────────────────────────────────
    "interior design ideas indian homes","living room decoration",
    "bedroom design tips","kitchen design modular",
    "bathroom renovation india","false ceiling design",
    "wallpaper vs paint","flooring options india",
    "sofa set price india","furniture online india",
    "home appliance brand comparison","best refrigerator india",
    "washing machine comparison","air conditioner best brand",
    "ceiling fan brand india","water purifier comparison",
    "inverter battery price","solar panel home",
    "water heater geyser","chimney for kitchen",
    "dishwasher worth buying india","microwave oven",
    "plants for home vastu","indoor plants benefits",
    "terrace garden ideas","balcony garden tips",
    "gardening tips india","composting at home",
    "declutter home tips","minimalist living",
    "feng shui home tips","vastu for bedroom",
    "vastu for kitchen","vastu for main door",
    "painting color for bedroom","color psychology home",
    "pet care india dogs","dog breed for india",
    "cat adoption india","fish aquarium setup",
    "pet food india brands","veterinary doctor near me",
    "fashion trends india 2024","ethnic wear online",
    "saree draping styles","lehenga buying guide",
    "kurta pajama style","sherwani for wedding",
    "wedding shopping budget","bridal makeup tips",
    "mehndi design latest","engagement ring diamond",
    "gold jewellery online","artificial jewellery",
    "makeup tutorial hindi","skincare routine indian",
    "hair care tips hindi","hair mask recipe home",
    "natural face pack","home remedies for glowing skin",
    "nail art designs simple","eyebrow threading tips",
    "baby care tips newborn","baby food first year",
    "toddler development stages","kids education apps",
    "parenting tips hindi","pregnancy diet indian",
    "prenatal yoga benefits","baby shower decoration",
    "senior citizen care india","old age home india",
    "retirement planning india","will kaise banaye",
 
    # ── SCIENCE & TECHNOLOGY (non-CS) ────────────────────────────────
    "climate change india effect","global warming solutions",
    "renewable energy india","solar energy future",
    "wind energy india","hydroelectric power",
    "nuclear energy india","electric vehicle india",
    "ev charging station india","tesla india launch",
    "tata nexon ev range","mg zs ev price",
    "hydrogen fuel cell car","hybrid car india",
    "petrol vs diesel vs electric","cng car benefits",
    "drone technology india","drone regulations india",
    "3d printing technology","nanotechnology uses",
    "quantum computing simple","augmented reality",
    "virtual reality headset","metaverse explained",
    "iot internet of things","smart home devices",
    "5g technology india","6g future plans",
    "satellite internet india","starlink india",
    "space tourism cost","isro satellite launch",
    "mars colonization future","moon mission 2025",
    "james webb space telescope","black hole image",
    "gravitational wave detection","dark matter explained",
    "string theory simple","multiverse theory",
    "human genome project","crispr gene editing",
    "stem cell therapy india","organ transplant india",
    "robotic surgery india","ai in medicine",
    "telemedicine india","health app india",
    "wearable health tech","apple watch health",
 
    # ── SOCIAL ISSUES & MISCELLANEOUS ────────────────────────────────
    "caste system india","reservation debate india",
    "women empowerment india","gender equality news",
    "lgbtq rights india","section 377 india",
    "child marriage india","dowry death news",
    "domestic violence helpline","women safety india",
    "child adoption india","foster care india",
    "ngo volunteer work india","social worker career",
    "environment protection news","plastic ban india",
    "river pollution cleanup","swachh bharat abhiyan",
    "open defecation free india","toilet facilities india",
    "smart city india progress","urban planning india",
    "rural development scheme","village development",
    "farmer protest india","msp kya hai farming",
    "crop insurance scheme","pm fasal bima yojana",
    "organic farming india","drip irrigation",
    "water conservation methods","rainwater harvesting",
    "air pollution solutions","noise pollution effects",
    "soil erosion prevention","deforestation news",
    "tiger conservation india","elephant corridor",
    "dolphin conservation","coral reef protection",
    "bird migration india","butterfly garden",
    "animal cruelty news","street dog problem",
    "cow slaughter law india","animal rights india",
    "child labor news","bonded labor news",
    "human trafficking india","drug abuse india",
    "mental health awareness","suicide prevention helpline",
    "old age problem india","loneliness epidemic",
    "income inequality india","poverty alleviation",
    "underprivileged education","midday meal scheme",
    "beti bachao beti padhao","girl child education",
}

NON_FITNESS_SINGLE = {
    "date","time","weather","news","movie","film","politics","coding",
    "programming","cricket","election","bitcoin","crypto","stock",
    "salary","recipe","astrology","horoscope","song","actor","actress",

     # Tech
    "coding","programming","algorithm","database","server","software",
    "hardware","internet","website","app","laptop","computer","smartphone",
    "tablet","gadget","technology","digital","cyber","robot","drone",
    "satellite","telescope","microscope","semiconductor","processor",
    # Finance
    "stock","share","market","trading","investment","portfolio","dividend",
    "bond","debenture","currency","dollar","euro","pound","yen","rupee",
    "gold","silver","commodity","derivative","futures","options",
    "insurance","premium","policy","claim","annuity","pension","retirement",
    # Entertainment
    "movie","film","series","show","episode","season","trailer","review",
    "actor","actress","director","producer","screenplay","sequel","prequel",
    "album","song","lyrics","concert","tour","festival","award","nomination",
    "netflix","hotstar","amazon","youtube","spotify","podcast","vlog","blog",
    # Sports
    "cricket","football","basketball","tennis","badminton","hockey",
    "kabaddi","wrestling","boxing","swimming","athletics","golf","rugby",
    "volleyball","handball","cycling","rowing","archery","shooting",
    "score","wicket","goal","point","match","tournament","league","cup",
    "trophy","medal","champion","runner","qualifier","semifinal","final",
    # News/Politics
    "election","vote","ballot","democracy","parliament","senate","congress",
    "president","minister","governor","mayor","councillor","mla","mp",
    "party","manifesto","policy","law","bill","act","amendment","ordinance",
    "protest","rally","strike","demonstration","agitation","march",
    "corruption","scam","scandal","controversy","accusation","verdict",
    "news","headline","breaking","update","report","journalism","media",
    # Religion
    "temple","mosque","church","gurudwara","monastery","shrine","altar",
    "prayer","worship","ritual","ceremony","festival","pilgrimage","fast",
    "mantra","hymn","sermon","scripture","bible","quran","gita","torah",
    "astrology","horoscope","zodiac","numerology","palmistry","tarot",
    "vastu","feng","shui","chakra","aura","meditation","spirituality",
    # Food/Cooking
    "recipe","cooking","baking","grilling","frying","boiling","steaming",
    "restaurant","cafe","dhaba","bakery","confectionery","catering",
    "biryani","curry","roti","naan","paratha","dosa","idli","samosa",
    "pizza","burger","sandwich","pasta","noodles","sushi","taco",
    "dessert","cake","pastry","cookie","candy","chocolate","ice cream",
    "spice","herb","sauce","gravy","marinade","seasoning","garnish",
    # Travel
    "tourism","travel","vacation","holiday","trip","journey","expedition",
    "hotel","hostel","resort","villa","cottage","campsite","homestay",
    "airport","railway","metro","highway","toll","bridge","tunnel",
    "visa","passport","immigration","customs","border","embassy","consulate",
    "map","route","navigation","distance","direction","location","place",
    # Education
    "school","college","university","institute","academy","coaching",
    "exam","test","assessment","assignment","project","thesis","dissertation",
    "degree","diploma","certificate","scholarship","fellowship","grant",
    "teacher","professor","lecturer","tutor","mentor","principal","dean",
    "syllabus","curriculum","textbook","library","laboratory","workshop",
    # Health (non-fitness)
    "doctor","physician","surgeon","specialist","dentist","optometrist",
    "hospital","clinic","pharmacy","laboratory","diagnostics","radiology",
    "medicine","tablet","capsule","injection","vaccine","antibiotic",
    "disease","illness","infection","virus","bacteria","fungi","parasite",
    "fever","pain","swelling","rash","allergy","inflammation","injury",
    "surgery","operation","procedure","therapy","treatment","recovery",
    # Misc
    "weather","climate","temperature","rainfall","humidity","pressure",
    "earthquake","flood","cyclone","tsunami","volcano","landslide","drought",
    "pollution","environment","ecology","biodiversity","conservation",
    "agriculture","farming","crop","harvest","irrigation","fertilizer",
    "animal","bird","reptile","insect","fish","mammal","amphibian","plant",
    "love","relationship","marriage","divorce","family","parenting","dating",
    "fashion","style","clothing","footwear","accessory","jewelry","cosmetic",
    "home","house","apartment","furniture","appliance","decoration","garden",
    "art","painting","sculpture","photography","music","dance","theater",
    "language","grammar","vocabulary","translation","interpretation",
    "psychology","sociology","philosophy","economics","anthropology",
    "law","legal","court","judge","lawyer","advocate","justice","rights",
}

def is_non_fitness(query: str) -> bool:
    q = query.lower().strip()
    for phrase in NON_FITNESS_KEYWORDS:
        if phrase in q:
            return True
    words = q.split()
    if len(words) <= 2:
        for w in words:
            clean = re.sub(r'[?!.,]', '', w)
            if clean in NON_FITNESS_SINGLE:
                return True
    return False

_dataset: list[dict] | None = None

def _load_dataset() -> list[dict]:
    global _dataset
    if _dataset is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "data", "gym_knowledge.json")
        with open(path, "r", encoding="utf-8-sig") as f:
            _dataset = json.load(f)
        print(f"[pipeline] Dataset: {len(_dataset)} entries loaded.")
    return _dataset

def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[!?.,।]+$', '', text).strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(.)\1+', r'\1', text)   # collapse hiii→hi
    return text

def _word_set(text: str) -> set:
    """Normalized word set — ignores word order for fuzzy matching."""
    stop = {'kya','hai','hain','ka','ki','ke','mein','se','ko','aur',
            'the','a','an','is','are','of','for','in','to','how','what',
            'when','why','which','does','do','can','should','i','me','my',
            'you','your','will','kaise','kitna','kitni','kitne','konsa'}
    words = set(_normalize(text).split())
    return words - stop

def dataset_match(query: str) -> str | None:
    data   = _load_dataset()
    q_norm = _normalize(query)
    q_words = _word_set(query)

    best_answer  = None
    best_score   = 0

    for entry in data:
        dq_norm  = _normalize(entry["question"])
        dq_words = _word_set(entry["question"])

        # Exact match
        if q_norm == dq_norm:
            print(f"[pipeline] ✅ Exact match: '{entry['question']}'")
            return entry["answer"]

        # Word-order independent matching — handles "workout best beginners ke liye"
        if len(q_words) >= 2 and len(dq_words) >= 2:
            overlap = len(q_words & dq_words)
            smaller = min(len(q_words), len(dq_words))
            score   = overlap / smaller if smaller > 0 else 0

            # High overlap ratio = good match regardless of word order
            if score >= 0.75 and overlap >= 2:
                if score > best_score:
                    best_score  = score
                    best_answer = entry["answer"]
                    print(f"[pipeline] 🔍 Word match: '{entry['question']}' score={score:.2f}")

    return best_answer

def get_threshold(query: str) -> float:
    words = query.strip().split()
    if len(words) == 1:   return 0.40
    if len(words) <= 3:   return 0.50
    return 0.70

def run_rag_pipeline(query: str) -> dict:
    if not query or not query.strip():
        return {"query": query, "retrieved_docs": [], "answer": "Please ask a fitness question."}

    print(f"\n{'='*60}\n[pipeline] Query: {query!r}\n{'='*60}")

    # Level 0: Non-fitness keyword pre-filter
    if is_non_fitness(query):
        print(f"[pipeline] ❌ Non-fitness keyword → instant reject.")
        return {"query": query, "retrieved_docs": [], "answer": OUT_OF_DOMAIN_ANSWER}

    # Level 1: Dataset match (word-order independent)
    direct = dataset_match(query)
    if direct:
        print(f"[pipeline] ⚡ Dataset match → direct answer.")
        return {"query": query, "retrieved_docs": [], "answer": direct}

    # Level 2: FAISS retrieval
    retriever      = get_retriever()
    retrieved_docs = retriever.retrieve(query, top_k=TOP_K)

    if not retrieved_docs:
        return {"query": query, "retrieved_docs": [], "answer": OUT_OF_DOMAIN_ANSWER}

    top_score = retrieved_docs[0]["score"]
    print(f"[pipeline] FAISS score: {top_score:.3f}")

    # Level 3: Too low → out of domain
    if top_score < OUT_OF_DOMAIN_THRESHOLD:
        print(f"[pipeline] ❌ Score too low → out of domain.")
        return {"query": query, "retrieved_docs": retrieved_docs, "answer": OUT_OF_DOMAIN_ANSWER}

    # Level 4: High score → direct FAISS answer
    threshold = get_threshold(query)
    if top_score >= threshold:
        print(f"[pipeline] ⚡ FAISS direct answer (score {top_score:.3f}).")
        return {"query": query, "retrieved_docs": retrieved_docs, "answer": retrieved_docs[0]["answer"]}

    # Level 5: LLM
    print(f"[pipeline] 🤖 LLM call.")
    generator = get_generator()
    answer    = generator.generate(query, retrieved_docs)
    return {"query": query, "retrieved_docs": retrieved_docs, "answer": answer}