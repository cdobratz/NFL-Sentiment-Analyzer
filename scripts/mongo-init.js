// MongoDB Initialization Script for NFL Sentiment Analyzer
// This script sets up the database schema, indexes, and initial data

// Switch to the NFL sentiment database
db = db.getSiblingDB('nfl_sentiment');

// Create collections with validation
db.createCollection('sentiments', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['text', 'sentiment', 'score', 'timestamp'],
            properties: {
                text: {
                    bsonType: 'string',
                    description: 'must be a string and is required'
                },
                sentiment: {
                    bsonType: 'string',
                    enum: ['POSITIVE', 'NEGATIVE', 'NEUTRAL'],
                    description: 'must be a valid sentiment label'
                },
                score: {
                    bsonType: 'double',
                    minimum: 0,
                    maximum: 1,
                    description: 'must be a number between 0 and 1'
                },
                timestamp: {
                    bsonType: 'date',
                    description: 'must be a date and is required'
                },
                team_id: {
                    bsonType: 'string',
                    description: 'optional team identifier'
                },
                player_id: {
                    bsonType: 'string',
                    description: 'optional player identifier'
                },
                game_id: {
                    bsonType: 'string',
                    description: 'optional game identifier'
                },
                source: {
                    bsonType: 'string',
                    enum: ['TWITTER', 'ESPN', 'NEWS', 'MANUAL'],
                    description: 'data source'
                }
            }
        }
    }
});

db.createCollection('users', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['email', 'username', 'password_hash', 'role', 'created_at'],
            properties: {
                email: {
                    bsonType: 'string',
                    pattern: '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                    description: 'must be a valid email address'
                },
                username: {
                    bsonType: 'string',
                    minLength: 3,
                    maxLength: 50,
                    description: 'must be a string between 3-50 characters'
                },
                password_hash: {
                    bsonType: 'string',
                    description: 'must be a hashed password'
                },
                role: {
                    bsonType: 'string',
                    enum: ['USER', 'ADMIN'],
                    description: 'must be a valid user role'
                },
                created_at: {
                    bsonType: 'date',
                    description: 'must be a date'
                },
                last_login: {
                    bsonType: 'date',
                    description: 'optional last login date'
                }
            }
        }
    }
});

db.createCollection('teams', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['name', 'abbreviation', 'conference', 'division'],
            properties: {
                name: {
                    bsonType: 'string',
                    description: 'team name is required'
                },
                abbreviation: {
                    bsonType: 'string',
                    minLength: 2,
                    maxLength: 5,
                    description: 'team abbreviation'
                },
                conference: {
                    bsonType: 'string',
                    enum: ['AFC', 'NFC'],
                    description: 'must be AFC or NFC'
                },
                division: {
                    bsonType: 'string',
                    enum: ['North', 'South', 'East', 'West'],
                    description: 'must be a valid division'
                }
            }
        }
    }
});

db.createCollection('games', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['home_team_id', 'away_team_id', 'game_date', 'season', 'week'],
            properties: {
                home_team_id: {
                    bsonType: 'string',
                    description: 'home team ID is required'
                },
                away_team_id: {
                    bsonType: 'string',
                    description: 'away team ID is required'
                },
                game_date: {
                    bsonType: 'date',
                    description: 'game date is required'
                },
                season: {
                    bsonType: 'int',
                    minimum: 2020,
                    description: 'season year'
                },
                week: {
                    bsonType: 'int',
                    minimum: 1,
                    maximum: 22,
                    description: 'week number'
                }
            }
        }
    }
});

// Create indexes for performance
print('Creating indexes...');

// Sentiments collection indexes
db.sentiments.createIndex({ 'timestamp': -1 });
db.sentiments.createIndex({ 'team_id': 1, 'timestamp': -1 });
db.sentiments.createIndex({ 'player_id': 1, 'timestamp': -1 });
db.sentiments.createIndex({ 'game_id': 1, 'timestamp': -1 });
db.sentiments.createIndex({ 'source': 1, 'timestamp': -1 });
db.sentiments.createIndex({ 'sentiment': 1, 'timestamp': -1 });
db.sentiments.createIndex({ 'text': 'text' }); // Text search index

// Users collection indexes
db.users.createIndex({ 'email': 1 }, { unique: true });
db.users.createIndex({ 'username': 1 }, { unique: true });
db.users.createIndex({ 'created_at': -1 });

// Teams collection indexes
db.teams.createIndex({ 'abbreviation': 1 }, { unique: true });
db.teams.createIndex({ 'conference': 1, 'division': 1 });

// Games collection indexes
db.games.createIndex({ 'game_date': -1 });
db.games.createIndex({ 'home_team_id': 1, 'game_date': -1 });
db.games.createIndex({ 'away_team_id': 1, 'game_date': -1 });
db.games.createIndex({ 'season': 1, 'week': 1 });

// Insert initial NFL teams data
print('Inserting NFL teams data...');

const nflTeams = [
    // AFC East
    { name: 'Buffalo Bills', abbreviation: 'BUF', conference: 'AFC', division: 'East' },
    { name: 'Miami Dolphins', abbreviation: 'MIA', conference: 'AFC', division: 'East' },
    { name: 'New England Patriots', abbreviation: 'NE', conference: 'AFC', division: 'East' },
    { name: 'New York Jets', abbreviation: 'NYJ', conference: 'AFC', division: 'East' },
    
    // AFC North
    { name: 'Baltimore Ravens', abbreviation: 'BAL', conference: 'AFC', division: 'North' },
    { name: 'Cincinnati Bengals', abbreviation: 'CIN', conference: 'AFC', division: 'North' },
    { name: 'Cleveland Browns', abbreviation: 'CLE', conference: 'AFC', division: 'North' },
    { name: 'Pittsburgh Steelers', abbreviation: 'PIT', conference: 'AFC', division: 'North' },
    
    // AFC South
    { name: 'Houston Texans', abbreviation: 'HOU', conference: 'AFC', division: 'South' },
    { name: 'Indianapolis Colts', abbreviation: 'IND', conference: 'AFC', division: 'South' },
    { name: 'Jacksonville Jaguars', abbreviation: 'JAX', conference: 'AFC', division: 'South' },
    { name: 'Tennessee Titans', abbreviation: 'TEN', conference: 'AFC', division: 'South' },
    
    // AFC West
    { name: 'Denver Broncos', abbreviation: 'DEN', conference: 'AFC', division: 'West' },
    { name: 'Kansas City Chiefs', abbreviation: 'KC', conference: 'AFC', division: 'West' },
    { name: 'Las Vegas Raiders', abbreviation: 'LV', conference: 'AFC', division: 'West' },
    { name: 'Los Angeles Chargers', abbreviation: 'LAC', conference: 'AFC', division: 'West' },
    
    // NFC East
    { name: 'Dallas Cowboys', abbreviation: 'DAL', conference: 'NFC', division: 'East' },
    { name: 'New York Giants', abbreviation: 'NYG', conference: 'NFC', division: 'East' },
    { name: 'Philadelphia Eagles', abbreviation: 'PHI', conference: 'NFC', division: 'East' },
    { name: 'Washington Commanders', abbreviation: 'WAS', conference: 'NFC', division: 'East' },
    
    // NFC North
    { name: 'Chicago Bears', abbreviation: 'CHI', conference: 'NFC', division: 'North' },
    { name: 'Detroit Lions', abbreviation: 'DET', conference: 'NFC', division: 'North' },
    { name: 'Green Bay Packers', abbreviation: 'GB', conference: 'NFC', division: 'North' },
    { name: 'Minnesota Vikings', abbreviation: 'MIN', conference: 'NFC', division: 'North' },
    
    // NFC South
    { name: 'Atlanta Falcons', abbreviation: 'ATL', conference: 'NFC', division: 'South' },
    { name: 'Carolina Panthers', abbreviation: 'CAR', conference: 'NFC', division: 'South' },
    { name: 'New Orleans Saints', abbreviation: 'NO', conference: 'NFC', division: 'South' },
    { name: 'Tampa Bay Buccaneers', abbreviation: 'TB', conference: 'NFC', division: 'South' },
    
    // NFC West
    { name: 'Arizona Cardinals', abbreviation: 'ARI', conference: 'NFC', division: 'West' },
    { name: 'Los Angeles Rams', abbreviation: 'LAR', conference: 'NFC', division: 'West' },
    { name: 'San Francisco 49ers', abbreviation: 'SF', conference: 'NFC', division: 'West' },
    { name: 'Seattle Seahawks', abbreviation: 'SEA', conference: 'NFC', division: 'West' }
];

db.teams.insertMany(nflTeams);

// Create admin user (password should be changed after first login)
print('Creating admin user...');
db.users.insertOne({
    email: 'admin@nfl-sentiment.com',
    username: 'admin',
    password_hash: '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj3QJgupsqHK', // 'admin123' - CHANGE THIS
    role: 'ADMIN',
    created_at: new Date(),
    last_login: null
});

// Create sample sentiment data for testing
print('Creating sample sentiment data...');
const sampleSentiments = [
    {
        text: 'The Chiefs are looking strong this season!',
        sentiment: 'POSITIVE',
        score: 0.85,
        timestamp: new Date(),
        team_id: 'KC',
        source: 'MANUAL'
    },
    {
        text: 'Not impressed with the Cowboys performance lately.',
        sentiment: 'NEGATIVE',
        score: 0.72,
        timestamp: new Date(),
        team_id: 'DAL',
        source: 'MANUAL'
    },
    {
        text: 'The game was okay, nothing special.',
        sentiment: 'NEUTRAL',
        score: 0.55,
        timestamp: new Date(),
        source: 'MANUAL'
    }
];

db.sentiments.insertMany(sampleSentiments);

print('Database initialization completed successfully!');
print('Collections created: ' + db.getCollectionNames().length);
print('Teams inserted: ' + db.teams.countDocuments());
print('Sample sentiments inserted: ' + db.sentiments.countDocuments());
print('Admin user created with username: admin (password: admin123 - PLEASE CHANGE)');