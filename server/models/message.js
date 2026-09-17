const mongoose = require("mongoose");

const messageSchema = new mongoose.Schema({

    visitor_id: {
        type: String,
        required: true
    },

    question: {
        type: String,
        required: true
    },

    answer: {
        type: String,
        required: true
    },

    createdAt: {
        type: Date,
        default: Date.now
    }

});

const Message = mongoose.model("Message", messageSchema);

module.exports = Message;