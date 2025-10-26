
require('dotenv').config();

const path = require('path');
const express = require('express');
const { MongoClient, ObjectId } = require('mongodb');

const app = express();              
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use(express.static(path.join(__dirname, 'public')));

// ---- ENV & constants ----
const MONGODB_URI = process.env.MONGODB_URI;
const DB_NAME      = process.env.DB_NAME;
const COLLECTION   = process.env.COLLECTION;
const PORT         = process.env.PORT || 5050;

if (!MONGODB_URI) {
  console.error('Missing MONGODB_URI in environment.');
  process.exit(1);
}

// ---- Lead states & legal transitions ----
const STATES = ['New', 'Contacted', 'In progress', 'Converted', 'Rejected'];
const ALLOWED_TRANSITIONS = {
  'New':        ['Contacted'],
  'Contacted':  ['In progress', 'Rejected'],
  'In progress':['Converted', 'Rejected'],
  'Converted':  [],
  'Rejected':   []
};
const canTransition = (from, to) => (ALLOWED_TRANSITIONS[from] || []).includes(to);

// ---- Mongo client ----
const client = new MongoClient(process.env.MONGODB_URI, {
    serverApi: { version: '1', strict: true, deprecationErrors: true },
    serverSelectionTimeoutMS: 10000,
  });

let col; // will hold the Mongo collection

// ---- Routes ----

// Health check
app.get('/health', (_req, res) => res.json({ ok: true }));

// States config for the UI
app.get('/api/config/states', (_req, res) => {
  res.json({ states: STATES, transitions: ALLOWED_TRANSITIONS });
});

// Fetch leads; by default, exclude Converted/Rejected from UI
app.get('/api/leads', async (req, res) => {
  try {
    const onlyActive = (req.query.active ?? 'true') === 'true';
    const limit = Math.min(parseInt(req.query.limit ?? '100', 10) || 100, 10000);

    const filter = {};


    // groupId and topK (rank within group)
    if (req.query.groupId) {
      const g = Number(req.query.groupId);
      if (!Number.isNaN(g)) filter.group_id = g;
    }
    const topK = Number(req.query.topK);
    if (!Number.isNaN(topK)) {
      filter.rank_within_group = { $lte: topK };
    }

    // Backfill any documents missing state -> "New" (one-time)
    await col.updateMany(
      { $or: [{ state: { $exists: false } }, { state: null }] },
      {
        $set: { state: 'New', updatedAt: new Date() },
        $push: { stateHistory: { state: 'New', at: new Date() } }
      }
    );

    if (onlyActive) {
      filter.state = { $nin: ['Converted', 'Rejected'] };
    }

    const docs = await col
      .find(filter)
      .sort({ relevance_score: -1 }) // keep ranking order
      .limit(limit)
      .toArray();

    // send _id as string for the frontend
    res.json(docs.map(d => ({ ...d, _id: d._id.toString() })));
  } catch (e) {
    console.error('GET /api/leads error:', e);
    res.status(500).json({ error: String(e) });
  }
});

// Update a lead's state with guarded transitions
app.put('/api/leads/:id/state', async (req, res) => {
  try {
    const { id } = req.params;
    const { toState } = req.body || {};

    if (!STATES.includes(toState)) {
      return res.status(400).json({ error: 'Invalid target state' });
    }

    const lead = await col.findOne({ _id: new ObjectId(id) });
    if (!lead) return res.status(404).json({ error: 'Lead not found' });

    const fromState = lead.state || 'New';
    if (!canTransition(fromState, toState)) {
      return res.status(400).json({ error: `Illegal transition: ${fromState} -> ${toState}` });
    }

    const now = new Date();
    await col.updateOne(
      { _id: new ObjectId(id) },
      {
        $set:  { state: toState, updatedAt: now },
        $push: { stateHistory: { state: toState, at: now } }
      }
    );

    const updated = await col.findOne({ _id: new ObjectId(id) });
    const removedFromUI = ['Converted', 'Rejected'].includes(toState); // hide in UI, don't delete in DB
    res.json({ lead: { ...updated, _id: updated._id.toString() }, removedFromUI });
  } catch (e) {
    console.error('PUT /api/leads/:id/state error:', e);
    res.status(500).json({ error: String(e) });
  }
});

// ---- Start server AFTER connecting to Mongo ----
(async () => {
  try {
    await client.connect();
    col = client.db(DB_NAME).collection(COLLECTION);
    app.listen(PORT, () => {
      console.log(`Server running on http://localhost:${PORT}`);
      console.log(`DB: ${DB_NAME}, Collection: ${COLLECTION}`);
    });
  } catch (e) {
    console.error('Failed to init server:', e);
    process.exit(1);
  }
})();
