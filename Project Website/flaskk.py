
from flask import Flask, render_template, request
import torch
import numpy as np


from transformers import T5Tokenizer
import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, LayerNormalization, Dropout, MultiHeadAttention
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Embedding, Concatenate, Attention, Masking
from tensorflow.keras.optimizers import RMSprop


    
app = Flask('__main__')
@app.route('/aboutus.html')
def about_us():
    return render_template('aboutus.html')

@app.route('/main.html')
def home():
    return render_template('main.html')



@app.route('/model.html')
def model1():
    return render_template('model.html')

@app.route("/model2.html")
def model_2():
    return render_template('model2.html')

@app.route("/model3.html")
def model_3():
    return render_template('model3.html')

@app.route("/index.html")
def dataset():
    return render_template('index.html')


@app.route('/')
def first():
    return render_template("main.html")

@app.route('/predictTransformer', methods = ['POST'])
def model_transformer():
    embedding_dim = 256
    vocab_size = 32128
    head_size = 64
    num_heads = 8
    ff_dim = 512
    dropout_rate = 0.1
    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout_rate):
        attention = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
        attention = Dropout(dropout_rate)(attention)
        attention = LayerNormalization(epsilon=1e-6)(attention + inputs)

        ff_output = Dense(ff_dim, activation='relu')(attention)
        ff_output = Dense(inputs.shape[-1])(ff_output)
        ff_output = Dropout(dropout_rate)(ff_output)
        encoder_output = LayerNormalization(epsilon=1e-6)(ff_output + attention)
        return encoder_output

    def transformer_decoder(inputs, enc_output, head_size, num_heads, ff_dim, dropout_rate):
        self_attention = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
        self_attention = Dropout(dropout_rate)(self_attention)
        self_attention = LayerNormalization(epsilon=1e-6)(self_attention + inputs)

        attention = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(self_attention, enc_output)
        attention = Dropout(dropout_rate)(attention)
        attention = LayerNormalization(epsilon=1e-6)(attention + self_attention)

        ff_output = Dense(ff_dim, activation='relu')(attention)
        ff_output = Dense(inputs.shape[-1])(ff_output)
        ff_output = Dropout(dropout_rate)(ff_output)
        decoder_output = LayerNormalization(epsilon=1e-6)(ff_output + attention)
        return decoder_output



    inputs_enc = Input(shape=(None,))
    inputs_dec = Input(shape=(None,))

    enc_emb = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs_enc)
    dec_emb = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs_dec)

    enc_output = transformer_encoder(enc_emb, head_size, num_heads, ff_dim, dropout_rate)
    dec_output = transformer_decoder(dec_emb, enc_output, head_size, num_heads, ff_dim, dropout_rate)

    final_output = Dense(vocab_size, activation='softmax')(dec_output)

    model_ = Model(inputs=[inputs_enc, inputs_dec], outputs=final_output)
    model_.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model_.load_model("ychennu_lkundan_ashaik5_transformer.h5")
    
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    def predict_sql(query, tokenizer, model, max_length=512):

        input_ids = tokenizer(
            query,
            return_tensors='np',
            padding='max_length',
            truncation=True,
            max_length=max_length
        )['input_ids']


        decoder_input = np.full((1, max_length), fill_value=tokenizer.pad_token_id)

        decoder_input[0, 0] = tokenizer.pad_token_id
        for i in range(1, max_length):

            predictions = model.predict([input_ids, decoder_input])[0, i-1, :]
            predicted_token_id = np.argmax(predictions)
            decoder_input[0, i] = predicted_token_id

            if predicted_token_id == tokenizer.eos_token_id:
                break
        predicted_sql = tokenizer.decode(decoder_input[0], skip_special_tokens=True)
        return predicted_sql
    data1 = request.form['inputText']
    predicted_sql_query = predict_sql(data1, tokenizer, model_, max_length=20)   
    return render_template("model.html", data = predicted_sql_query)


@app.route('/predictLSTM', methods = ['POST'])
def model2_prediction():
    data1 = request.form['inputText']

    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    vocab_size = tokenizer.vocab_size
    embedding_dim = 256
    latent_dim = 512


    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)


    encoder_inputs = Input(shape=(None,), dtype='int32', name='encoder_inputs')
    encoder_embedded = embedding_layer(encoder_inputs)
    encoder_gru = GRU(latent_dim, return_sequences=True, return_state=True, name='encoder_gru')
    encoder_outputs, state_h = encoder_gru(encoder_embedded)


    decoder_inputs = Input(shape=(None,), dtype='int32', name='decoder_inputs')
    decoder_embedded = embedding_layer(decoder_inputs)
    decoder_gru = GRU(latent_dim, return_sequences=True, return_state=True, name='decoder_gru')
    decoder_outputs, _ = decoder_gru(decoder_embedded, initial_state=state_h)


    attention_layer = Attention(name='attention_layer')
    attention_result = attention_layer([decoder_outputs, encoder_outputs])


    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attention_result])


    decoder_dense = Dense(vocab_size, activation='softmax', name='output_layer')
    decoder_outputs = decoder_dense(decoder_concat_input)


    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


    model.compile(optimizer=RMSprop(learning_rate=0.001), loss='sparse_categorical_crossentropy')
    model.load_weights('ychennu_lkundan_ashaik5_lstm.h5')
    def preprocess_input(sentence, tokenizer, max_length):

        tokenized_input = tokenizer(
            sentence,
            return_tensors='np',
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
        return tokenized_input['input_ids']
    def predict_sql(input_sentence, tokenizer, model, max_length):

        input_ids = preprocess_input(input_sentence, tokenizer, max_length)


        start_token_id = tokenizer.pad_token_id
        decoder_input = np.array([[start_token_id]])


        output_tokens = []

        for _ in range(max_length):

            current_pred = model.predict([input_ids, decoder_input])
            next_token_id = np.argmax(current_pred[0, -1, :], axis=-1)


            output_tokens.append(next_token_id)


            decoder_input = np.hstack([decoder_input, [[next_token_id]]])


            if next_token_id == tokenizer.eos_token_id:
                break


        predicted_query = tokenizer.decode(output_tokens, skip_special_tokens=True)
        return predicted_query
    value = predict_sql(data1, tokenizer, model, max_length=20)
    return render_template("model2.html", data = value)



@app.route('/predictGRU', methods = ['POST'])
def model3_transformer():

    vocab_size = tokenizer.vocab_size
    embedding_dim = 256
    latent_dim = 512


    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)


    encoder_inputs = Input(shape=(None,), dtype='int32', name='encoder_inputs')
    encoder_embedded = embedding_layer(encoder_inputs)
    encoder_gru = GRU(latent_dim, return_sequences=True, return_state=True, name='encoder_gru')
    encoder_outputs, state_h = encoder_gru(encoder_embedded)

        
    decoder_inputs = Input(shape=(None,), dtype='int32', name='decoder_inputs')
    decoder_embedded = embedding_layer(decoder_inputs)
    decoder_gru = GRU(latent_dim, return_sequences=True, return_state=True, name='decoder_gru')
    decoder_outputs, _ = decoder_gru(decoder_embedded, initial_state=state_h)


    attention_layer = Attention(name='attention_layer')
    attention_result = attention_layer([decoder_outputs, encoder_outputs])


    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attention_result])


    decoder_dense = Dense(vocab_size, activation='softmax', name='output_layer')
    decoder_outputs = decoder_dense(decoder_concat_input)


    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


    model.compile(optimizer=RMSprop(learning_rate=0.001), loss='sparse_categorical_crossentropy')
    model = load_model('ychennu_lkundan_ashaik5_GRU.h5')
    
    def preprocess_input(sentence, tokenizer, max_length):

        tokenized_input = tokenizer(
            sentence,
            return_tensors='np',
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
        return tokenized_input['input_ids']
    def predict_sql(input_sentence, tokenizer, model, max_length):

        input_ids = preprocess_input(input_sentence, tokenizer, max_length)


        start_token_id = tokenizer.pad_token_id
        decoder_input = np.array([[start_token_id]])


        output_tokens = []

        for _ in range(max_length):

            current_pred = model.predict([input_ids, decoder_input])
            next_token_id = np.argmax(current_pred[0, -1, :], axis=-1)


            output_tokens.append(next_token_id)


            decoder_input = np.hstack([decoder_input, [[next_token_id]]])


            if next_token_id == tokenizer.eos_token_id:
                break


        predicted_query = tokenizer.decode(output_tokens, skip_special_tokens=True)
        return predicted_query
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    data1 = request.form['inputText']
    predicted_sql_query = predict_sql(data1, tokenizer, model, max_length=20)
    return render_template("model3.html",data = predicted_sql_query)


if __name__ == '__main__':
    app.run(debug=True)
    


