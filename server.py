
from flask import Flask, request, jsonify
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Map watermark codes to actions
ACTION_MAP = {
    "101101": "PLAY_COMMERCIAL_SPOT_01",
    "011010": "LOG_USER_INTERACTION_EVENT",
    "111000": "CHANGE_LIGHTING_SCENE_TO_BLUE",
    "000111": "TRIGGER_ALARM_SYSTEM",
    "110011": "START_VIDEO_RECORDING",
    "100100": "SEND_NOTIFICATION_ALERT",
}

@app.route("/trigger", methods=['POST'])
def handle_trigger():
    """Receive watermark detection event."""
    try:
        data = request.get_json()
        
        if not data or 'code' not in data:
            logging.warning("⚠ Invalid request - missing 'code' field")
            return jsonify({
                "status": "error",
                "message": "Missing 'code' field"
            }), 400
        
        received_code = data['code']
        logging.info(f"📥 Received watermark code: {received_code}")
        
        # Look up action
        action = ACTION_MAP.get(received_code, "UNKNOWN_CODE")
        
        if action != "UNKNOWN_CODE":
            logging.info(f"🎬 EXECUTING ACTION: {action}")
            # Here you would actually trigger the action
            # For now, just logging it
        else:
            logging.warning(f"⚠ Unknown code '{received_code}' - no action defined")
        
        return jsonify({
            "status": "success",
            "received_code": received_code,
            "action": action
        })
        
    except Exception as e:
        logging.error(f"❌ Server error: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": "Internal server error"
        }), 500

@app.route("/health", methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "actions_available": len(ACTION_MAP)
    })

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 WATERMARK ACTION SERVER STARTING")
    print(f"   Available actions: {len(ACTION_MAP)}")
    print(f"   Listening on: http://0.0.0.0:5000")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
