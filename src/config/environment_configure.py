
mode = "local"

if mode=='local':#测试环境
    NEO4J_IP = "172.18.82.65"
    NEO4J_PORT = 7474
    NEO4J_USER_NAME = "neo4j"
    NEO4J_PASSWORD = "neo4j"

    MONGO_IP="172.18.82.78", 
    MONGO_PORT=27017
    MONGO_DB="apb_crawler"
    
    SERVER_IP = "172.18.82.65"
    SERVER_PORT = 1242
    
    NLP_PLATFORM_IP = '172.18.89.13'
    NLP_PLATFORM_PORT = 1240
    
    #需要查询mysql来获取地名数据
    MYSQL_IP = '172.18.82.35'
    MYSQL_PORT = 3306
    MYSQL_DB = 'common_tag'
    MYSQL_USER = 'root'
    MYSQL_PASSWORD = 'Founder123'
    MYSQL_CHATSET = 'utf8'
    MYSQL_DB_DAOGU = 'daogu_prod'#捣鼓精炼坊数据db
    
elif mode=="test":
    NEO4J_IP = "172.18.82.252"
    NEO4J_PORT = 7474
    NEO4J_USER_NAME = "neo4j"
    NEO4J_PASSWORD = "neo4j2019"
    
    MONGO_IP="172.18.82.78", 
    MONGO_PORT=27017
    MONGO_DB="apb_crawler"
    
    SERVER_IP = "172.18.89.13"
    SERVER_PORT = 1242
    
elif mode=='11':
    NEO4J_IP = "172.18.82.252"
    NEO4J_PORT = 7474
    NEO4J_USER_NAME = "neo4j"
    NEO4J_PASSWORD = "neo4j2019"
    
    MONGO_IP="172.18.82.78", 
    MONGO_PORT=27017
    MONGO_DB="apb_crawler"
    
    SERVER_IP = "172.18.89.11"
    SERVER_PORT = 1242
	
    NLP_PLATFORM_IP = '172.18.89.13'
    NLP_PLATFORM_PORT = 1240
    
elif mode=='home':
    NLP_PLATFORM_IP = '192.168.1.201'
    NLP_PLATFORM_PORT = 1240
    
    NEO4J_IP = "172.18.82.252"
    NEO4J_PORT = 7474
    NEO4J_USER_NAME = "neo4j"
    NEO4J_PASSWORD = "neo4j2019"
    
    MONGO_IP="172.18.82.78", 
    MONGO_PORT=27017
    MONGO_DB="apb_crawler"
    
    SERVER_IP = "192.168.1.100"
    SERVER_PORT = 1242