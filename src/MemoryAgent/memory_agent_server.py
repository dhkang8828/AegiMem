"""
Memory Agent REST API Server (GNR-SP)

Runs on GNR-SP (Linux server) and executes memory tests via C library.
Communicates with RL Agent (Windows PC) via REST API.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import argparse
from typing import Dict, Any

from memory_agent_c_wrapper import MemoryAgentC, OperationType


app = Flask(__name__)
CORS(app)  # Enable CORS for Windows PC client

# Global memory agent instance
memory_agent: MemoryAgentC = None


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint

    Returns:
        JSON: {
            "status": "healthy",
            "initialized": bool
        }
    """
    return jsonify({
        'status': 'healthy',
        'initialized': memory_agent.is_initialized() if memory_agent else False
    })


@app.route('/execute_action', methods=['POST'])
def execute_action():
    """
    Execute memory test action

    Request JSON:
        {
            "action": int (0-1535)
        }

    Response JSON:
        {
            "success": true,
            "ce_detected": bool,
            "ce_volatile": int,
            "ce_persistent": int,
            "ce_total": int,
            "temperature": int,
            "operation": str,
            "pattern": str
        }
    """
    try:
        data = request.get_json()
        action = data.get('action')

        if action is None:
            return jsonify({'error': 'Missing action parameter'}), 400

        if not (0 <= action < 1536):
            return jsonify({'error': 'Invalid action (must be 0-1535)'}), 400

        # Execute action via C library
        logging.info(f"Executing action {action}")
        ce_info, success = memory_agent.execute_action(action)

        # Decode action for response
        operation_type, pattern = memory_agent.decode_action(action)

        result = {
            'success': success,
            'ce_detected': ce_info.has_errors(),
            'ce_volatile': ce_info.volatile_count,
            'ce_persistent': ce_info.persistent_count,
            'ce_total': ce_info.total_count,
            'temperature': ce_info.temperature,
            'operation': OperationType.name(operation_type),
            'pattern': f'0x{pattern:02X}'
        }

        logging.info(f"Action {action} completed: CE={ce_info.total_count}")
        return jsonify(result)

    except Exception as e:
        logging.error(f"Action execution error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/reset_baseline', methods=['POST'])
def reset_baseline():
    """
    Reset CE baseline (call at start of episode)

    Returns:
        JSON: {"success": bool}
    """
    try:
        success = memory_agent.reset_baseline()
        return jsonify({'success': success})
    except Exception as e:
        logging.error(f"Reset baseline error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/get_ce_info', methods=['GET'])
def get_ce_info():
    """
    Get current CE information without executing test

    Returns:
        JSON: {
            "volatile_count": int,
            "persistent_count": int,
            "total_count": int,
            "temperature": int
        }
    """
    try:
        ce_info = memory_agent.get_ce_info()
        return jsonify({
            'volatile_count': ce_info.volatile_count,
            'persistent_count': ce_info.persistent_count,
            'total_count': ce_info.total_count,
            'temperature': ce_info.temperature,
            'health_status': ce_info.health_status
        })
    except Exception as e:
        logging.error(f"Get CE info error: {e}")
        return jsonify({'error': str(e)}), 500


def main():
    """Main entry point for Memory Agent server"""
    parser = argparse.ArgumentParser(
        description='Memory Agent REST API Server for GNR-SP',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--host', default='0.0.0.0',
                        help='Host address (0.0.0.0 for all interfaces)')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port number')
    parser.add_argument('--devdax', default='/dev/dax0.0',
                        help='devdax device path')
    parser.add_argument('--memory-size', type=int, default=1024,
                        help='Memory size in MB (entire memory will be tested)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize global memory agent
    global memory_agent
    memory_agent = MemoryAgentC()

    logging.info("="*60)
    logging.info("Memory Agent REST API Server")
    logging.info("="*60)
    logging.info(f"Server: {args.host}:{args.port}")
    logging.info(f"devdax: {args.devdax}")
    logging.info(f"Memory size: {args.memory_size} MB")
    logging.info(f"Test coverage: 100% (entire memory)")
    logging.info("="*60)

    # Initialize Memory Agent C library
    try:
        logging.info("Initializing Memory Agent C library...")
        memory_agent.init(
            devdax_path=args.devdax,
            memory_size_mb=args.memory_size
        )
        logging.info("Memory Agent initialized successfully!")
    except Exception as e:
        logging.error(f"Failed to initialize Memory Agent: {e}")
        logging.error("Server will start but actions will fail until initialized")

    # Run Flask app
    logging.info(f"\nStarting server on {args.host}:{args.port}...")
    logging.info("Ctrl+C to stop\n")

    try:
        app.run(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        logging.info("\nShutting down...")
    finally:
        memory_agent.cleanup()
        logging.info("Memory Agent cleaned up. Goodbye!")


if __name__ == '__main__':
    main()
