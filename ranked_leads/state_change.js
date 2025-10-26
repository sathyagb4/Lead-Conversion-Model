// Watches results_new for state changes, writes labeled docs to leads_feedback,
// and triggers retrain if labeled set grew by ≥5% since last training.

require('dotenv').config();
const { MongoClient } = require('mongodb');
const { spawn } = require('child_process');

const MONGODB_URI = process.env.MONGODB_URI;
const DB_NAME     = process.env.DB_NAME;
const COLLECTION  = process.env.COLLECTION;
const FEEDBACK    = process.env.FEEDBACK_COLL;
const META_COLL   = process.env.META_COLL;
const MODEL_NAME  = process.env.MODEL_NAME;
const PYTHON_BIN  = process.env.PYTHON_BIN;
const RETRAIN_CMD = process.env.RETRAIN_CMD;
const NEW_DATA_THRESHOLD = Number(process.env.NEW_DATA_THRESHOLD || 0.05);

(async () => {
  const client = new MongoClient(MONGODB_URI);
  await client.connect();
  const db = client.db(DB_NAME);
  const results = db.collection(COLLECTION);
  const feedback = db.collection(FEEDBACK);
  const meta = db.collection(META_COLL);

  console.log('[watcher] Watching for state changes in', COLLECTION);

  //checks for converted or rejected 
  const pipeline = [
    { $match: { operationType: 'update' } },
    { $match: { 'updateDescription.updatedFields.state': { $in: ['converted','rejected','Converted','Rejected'] } } }

  ];

  const changeStream = results.watch(pipeline, { fullDocument: 'updateLookup' });

  changeStream.on('change', async (evt) => {
    const doc = evt.fullDocument;
    const state = (doc.state || '').toLowerCase();
    if (!['converted', 'rejected'].includes(state)) return;

    const label = state === 'converted' ? 1 : 0;
    await feedback.updateOne(
      { lead_id: doc.lead_id },
      { $set: { ...doc, label, labeled_at: new Date().toISOString() } },
      { upsert: true }
    );
    console.log(`[watcher] Feedback added: ${doc.lead_id} = ${state}`);

    const count = await feedback.countDocuments({});
    const metaDoc = await meta.findOne({ model_name: MODEL_NAME }) || {};
    const lastCount = metaDoc.last_trained_count || 0;
    const grew = count >= Math.max(1, lastCount) * (1 + NEW_DATA_THRESHOLD);

    if (grew || lastCount === 0) {
      console.log(`[watcher] Growth threshold met (${count} vs ${lastCount}), retraining...`);
      const retrain = spawn(PYTHON_BIN, [RETRAIN_CMD, '--push'], { stdio: 'inherit' });
      retrain.on('close', code => console.log(`[watcher] Retrain job exited with code ${code}`));
    }
  });
})();
   
