function set_encoding() {
  $('table-attrs-list li').click(function (e) {
  });
}

function set_selectable() {
  $('#input-cols-list li').click(function (e) {
      e.preventDefault()
      $that = $(this);

      if ($(this).hasClass('active')) {
        $(this).removeClass('active');
      }
      else {
        $that.addClass('active');
      }
    });
}

function load_data() {

  var data =
    {
      path_to_data: path_to_data
    };

  var dataToSend = JSON.stringify(data);

  $("#input-cols-list li").each(function (index) {
    $(this).remove();
  });
  
  $.ajax({
    url: '/open_file',
    type: 'POST',
    data: dataToSend,
    processData: false,
    contentType: false,

    success: function (jsonResponse) {
      var objresponse = JSON.parse(jsonResponse);
      table_loaded = objresponse;
      path_to_data = objresponse['path_to_data'];
      if (objresponse['msg'] == 'success') {
        
        var if_number = objresponse['if_number'];
        var attr_content = $("<ul>");
        attr_content.attr("id", "table-attrs-list");
        attr_content.addClass("list-group");
        // attr_content = '<ul class="list-group" id="table-attrs-list">'
        for (var key in if_number) {
          if (if_number[key] == true) {
            //cols_type[key] = 'Number';
            var list_number = $("<li>", {
              class: "list-group-item",
              html: "<p class='text-right'><b>" + key + "</b> <span class='label label-primary'>Number</span></p>"
            });
            list_number.attr("type", "Number");
            //attr_content += `
            //<li class="list-group-item"><p class="text-right"><b>`+ key + `</b> <span class="label label-primary">Number</span></p></li>
            //`;
            attr_content.append(list_number);
          }
          else {
            //cols_type[key] = 'String';
            var list_string = $("<li>", {
              class: "list-group-item"
            });
            var paragraph = $("<p>", {
              class: "text-right",
              html: "<b>" + key + " </b>"
            });
            var label = $("<span>", {
              class: "label label-danger"
            }).text("String");
            paragraph.append(label);
            list_string.append(paragraph);
            list_string.attr("type", "String");

            var ordinal_button = $("<button>");
            ordinal_button.html("<b>Ordinal</b>");
            ordinal_button.attr("id", "ordinal_button");
            ordinal_button.addClass("btn btn-default btn-sm");
            ordinal_button.click(function() {
              list_string.attr("type", "Ordinal");
              label.removeClass("label-danger");
              label.addClass("label-primary");
              $(this).css("background-color", "#5e9dd4");
              $(this).parent().find("#onehot_button").css("background-color", "#ffffff");
              if ($(this).parent().hasClass("onehot")) {
                $(this).parent().removeClass("onehot");
              }
              $(this).parent().addClass("ordinal");
            })

            var onehot_button = $("<button>");
            onehot_button.html("<b>One-hot</b>");
            onehot_button.attr("id", "onehot_button");
            onehot_button.addClass("btn btn-default btn-sm");
            onehot_button.click(function() {
              list_string.attr("type", "One-hot");
              label.removeClass("label-danger");
              label.addClass("label-primary");
              //cols_type[key] = 'Encoded';
              $(this).css("background-color", "#5e9dd4");
              $(this).parent().find("#ordinal_button").css("background-color", "#ffffff");
              if ($(this).parent().hasClass("ordinal")) {
                $(this).parent().removeClass("ordinal");
              }
              $(this).parent().addClass("onehot");
            })
            //onehot_button.attr("id", "asdf");
            list_string.append(ordinal_button);
            list_string.append(onehot_button);
            attr_content.append(list_string);
          }
        }
        // attr_content += '</ul>'
        $('#table-attrs').html(attr_content);
        
        $(function () {
          console.log('ready');
          
          $('#table-attrs-list li').click(function (e) {
            e.preventDefault()

            $that = $(this);
            if ($(this).hasClass('active')) {
              $(this).removeClass('active');
            }
            else {
              $that.addClass('active');
            }
          });
        });
        
      }
      
      else {
        $('#modal-title').html('Error');
        $('#modal-content').html('<div class="alert alert-danger" role="alert"> Something went wrong. Error code=' + objresponse['msg'] + '</div>');
        $('#my-modal').modal('show');
        
        $('#my-modal').on('hidden.bs.modal', function (e) {
          window.location = "/";
        });
        
      }
      
    },
    error: function (jsonResponse) {
      alert('Something went wrong.');
    }
    
    
  });
  
  if (input_cols.length > 0) {
    for (i in input_cols) {
      attr_content = `
      <li class="list-group-item"><p class="text-right"><b>`+ input_cols[i] + `</b> <span class="label label-primary">Number</span></p></li>
      `;
      $('#input-cols-list').append(attr_content);
      $('#input-cols').html('');
    }
  }
  
  
  set_selectable();
  
  if (target_col != "null" && target_col!=null) {
    target_col_html = `
    <p class="text-center"><b>`+ target_col + `</b> <span class="label label-primary">Number</span></p>        `
    $('#target-col').html(target_col_html);
  }
  else {
    target_col = null;
  }
};

function load_data_to_table() {

  var data =
    {
      path_to_data: path_to_data
    };

  var dataToSend = JSON.stringify(data);

  $("#input-cols-list li").each(function (index) {
    $(this).remove();
  });

  $.ajax({
    url: '/open_file',
    type: 'POST',
    data: dataToSend,
    processData: false,
    contentType: false,

    success: function (jsonResponse) {
      var objresponse = JSON.parse(jsonResponse);
      table_loaded = objresponse;

      path_to_data = objresponse['path_to_data'];
      if (objresponse['msg'] == 'success') {

        header_html = ''
        for (i = 0; i < objresponse['header'].length; i++) {
          header_html += '<th>' + objresponse['header'][i] + '</th>'
        }

        table_html = `
                  <table id="csv-table" class="display" cellspacing="0" width="100%" height="100%">
                  <thead>
                    <tr>
                      `+ header_html + `
                    </tr>
                  </thead>
                </table>
                `

        $('#table-content').html(table_html);

        try {
          $('#csv-table').dataTable().fnDestroy();
        }
        catch (err) {
          alert(err);
        }

        $('#csv-table').DataTable({
          "scrollX": true,
          "scrollY": "200px",
          "scrollCollapse": true,
          "paging": true
        });

        $('#csv-table').dataTable().fnClearTable();

        for (i = 0; i < objresponse['rows'].length; i++) {
          row = objresponse['rows'][i];
          $('#csv-table').dataTable().fnAddData(row);
        }

        var if_number = objresponse['if_number'];

        attr_content = '<ul class="list-group" id="table-attrs-list">'
        for (var key in if_number) {
          if (if_number[key] == true) {
            attr_content += `
                      <li class="list-group-item"><p class="text-right"><b>`+ key + `</b> <span class="label label-primary">Number</span></p></li>
                    `;
            //cols_type[key] = 'Number';
          }
          else {
            attr_content += `
                      <li class="list-group-item"><p class="text-right"><b>`+ key + `</b> <span class="label label-danger">String</span></p>
                      <button id="select-all" data-toggle="tooltip" title="Select all" type="button"
                      class="btn btn-default btn-sm">
                      Test
                      </button>
                      <button id="select-all" data-toggle="tooltip" title="Select all" type="button"
                      class="btn btn-default btn-sm">asdf
                    </button></li>
                    `;
            //cols_type[key] = 'String';
          }

        }
        attr_content += '</ul>'
        $('#table-attrs').html(attr_content);

        $(function () {
          console.log('ready');

          $('#table-attrs-list li').click(function (e) {
            e.preventDefault()

            $that = $(this);

            if ($(this).hasClass('active')) {
              $(this).removeClass('active');
            }
            else {
              $that.addClass('active');
            }
          });
        });

        if (input_cols.length > 0) {
          for (i in input_cols) {
            attr_content = `
                    <li class="list-group-item"><p class="text-right"><b>`+ input_cols[i] + `</b> <span class="label label-primary">Number</span></p></li>
                  `;
            $('#input-cols-list').append(attr_content);
            $('#input-cols').html('');
          }
        }

         $('#input-cols-list li').click(function (e) {
            e.preventDefault()

            $that = $(this);

            if ($(this).hasClass('active')) {
              $(this).removeClass('active');
            }
            else {
              $that.addClass('active');
            }
          });

        if (target_col != "null" && target_col!=null) {
          target_col_html = `
              <p class="text-center"><b>`+ target_col + `</b> <span class="label label-primary">Number</span></p>
            `
          $('#target-col').html(target_col_html);
        }
        else {
          target_col = null;
        }

      }

      else {
        $('#modal-title').html('Error');
        $('#modal-content').html('<div class="alert alert-danger" role="alert"> Something went wrong. Error code=' + objresponse['msg'] + '</div>');
        $('#my-modal').modal('show');

        $('#my-modal').on('hidden.bs.modal', function (e) {
          window.location = "/";
        });

      }

    },
    error: function (jsonResponse) {
      alert('Something went wrong.');
    }


  });

};


function clear_table() {

  $('#table-content').html(`
      <div class="alert alert-info" role="alert">
        Please open a CSV file.
      </div>
      `);

  $('#table-attrs').html(`<div class="alert alert-info" role="alert">
        No contents to display.
      </div>`);

  $('#input-cols').html(`<div class="alert alert-info" role="alert">
        No input column has been selected.
      </div>`);

  $('#target-col').html(`<div class="alert alert-info" role="alert">
        No target column has been selected.
      </div>`);

  $('#corr-chart').html(``);

  $('#avail-chart-list').html(`<select class="form-control">
              <option>Chart not available.</option>
            </select>`);

  table_loaded = false;
  target_col = null;
  path_to_data = '';
  //cols_type = {};
  input_cols = [];

}


String.prototype.unescapeHtml = function () {
  var temp = document.createElement("div");
  temp.innerHTML = this;
  var result = temp.childNodes[0].nodeValue;
  temp.removeChild(temp.firstChild);
  return result;
}

function add_to_target() {
  var col_to_add = [];
  var col_to_add_type = [];

  $("#table-attrs-list li").each(function (index) {
    if ($(this).hasClass('active')) {
      var last_index = $(this).text().lastIndexOf(" ");
      var input_col_key = $(this).text().substring(0, last_index);
      var input_col_type = $(this).attr("type");
      col_to_add.push(input_col_key);
      col_to_add_type.push(input_col_type);
      $(this).removeClass('active');
    }
  });
  if (col_to_add.length == 1) {
    if (col_to_add_type[0] == 'Number') {
      target_col = col_to_add[0];
      //cols_type[col_to_add[0]] = col_to_add_type[0];
      var target_col_jq = $("<p>", {
        html: "<p class='text-center'><b>"+ col_to_add[0] + "</b> <span class='label label-primary'>Number</span></p>"
      });
      target_col_html = $(target_col_jq).html();
      $('#target-col').html(target_col_jq);
      return;
    }
    if (col_to_add_type[0] == "String") {
      $('#modal-title').html('Warning');
      $('#modal-content').html('<div class="alert alert-danger" role="alert"> ' + 'If target column is a String, then it must be ordinal encoded.' + '</div>');
      $('#my-modal').modal('show');
      return;
    }
    if (col_to_add_type[0] == "Ordinal") {
      target_col = col_to_add[0];
      ordinal_cols.push(col_to_add[0]);
      //cols_type[col_to_add[0]] = col_to_add_type[0];
      var target_col_jq = $("<p>", {
        class: "ordinal text-center",
        html: "<b>"+ col_to_add[0] + "</b> <span class='label label-primary'>Ordinal</span>"
      });
      target_col_html = $(target_col_jq).html();
      $('#target-col').html(target_col_jq);
      return;
    }
    // One-hot
    $('#modal-title').html('Warning');
    $('#modal-content').html('<div class="alert alert-danger" role="alert"> ' + 'Target column can only be ordinal encoded, not one-hot.' + '</div>');
    $('#my-modal').modal('show');
  }
  else {
    $('#modal-title').html('Warning');
    $('#modal-content').html('<div class="alert alert-danger" role="alert"> ' + 'Please select exactly one column for target.' + '</div>');
    $('#my-modal').modal('show');
  }
};

function add_to_input() {
  var not_added_list = []
  $("#table-attrs-list li").each(function (index) {

    if ($(this).hasClass('active')) {

      var last_index = $(this).text().lastIndexOf(" ");
      var input_col_key = $(this).text().substring(0, last_index);
      var input_col_type = $(this).attr("type");
      if ($.inArray(input_col_key, input_cols) == -1) {
        if (input_col_type == 'String') {
          not_added_list.push(input_col_key);
        }
        else if (input_col_type == "Number") {
          input_cols.push(input_col_key);
          attr_content = `
                    <li class="list-group-item Number"><p class="text-right"><b>`+ input_col_key + `</b> <span class="label label-primary">Number</span></p></li>
                  `;
          $('#input-cols-list').append(attr_content);
          $('#input-cols').html('');
        }
        else if (input_col_type == "Ordinal") {
          input_cols.push(input_col_key);
          ordinal_cols.push(input_col_key);
          attr_content = `
                    <li class="list-group-item Ordinal"><p class="text-right"><b>`+ input_col_key + `</b> <span class="label label-primary">Ordinal</span></p></li>
                  `;
          $('#input-cols-list').append(attr_content);
          $('#input-cols').html('');
        }
        else if (input_col_type == "One-hot") {
          input_cols.push(input_col_key);
          attr_content = `
                    <li class="list-group-item One-hot"><p class="text-right"><b>`+ input_col_key + `</b> <span class="label label-primary">One-hot</span></p></li>
                  `;
          $('#input-cols-list').append(attr_content);
          $('#input-cols').html('');
        }
        else {
          console.log("ERROR: unrecognized input_col_type");
        }
      }; // TODO: if encoded col is already in input, make it so you can input it again with a different encoding

      $(this).removeClass('active');
    }
  });

  if (not_added_list.length > 0) {
    $('#modal-title').html('Warning');
    $('#modal-content').html('<div class="alert alert-warning" role="alert"> ' + 'String type columns [' + not_added_list + '] cannot be added to the input column list. Please choose an encoding first.' + '</div>');
    $('#my-modal').modal('show');
  }

  //attr_content+= ''


  if (input_cols.length == 0) {
    $('#input-cols').html(`<div class="alert alert-info" role="alert">
            No input column has been selected.
          </div>`);
  }
  $(function () {
    console.log('ready');

    $('#input-cols-list li').click(function (e) {
      e.preventDefault()

      $that = $(this);

      if ($(this).hasClass('active')) {
        $(this).removeClass('active');
      }
      else {
        $that.addClass('active');
      }
    });
  });
  
}

function select_all_input_cols() {
  $("#input-cols-list li").each(function (index) {
    $(this).addClass('active');
  });
}

function unselect_all_input_cols() {
  $("#input-cols-list li").each(function (index) {
    if ($(this).hasClass('active')) {
      $(this).removeClass('active');
    }
  });
}

function unselect_all() {
  $("#table-attrs-list li").each(function (index) {
    if ($(this).hasClass('active')) {
      $(this).removeClass('active');
    }
  });
}

function select_all() {
  $("#table-attrs-list li").each(function (index) {
    $(this).addClass('active');
  });
}

function remove_input_cols() {
  $("#input-cols-list li").each(function (index) {

    if ($(this).hasClass('active')) {
      var last_index = $(this).text().lastIndexOf(" ");
      var input_col_key = $(this).text().substring(0, last_index);
      input_cols.splice(input_cols.indexOf(input_col_key), 1);
      ordinal_cols.splice(ordinal_cols.indexOf(input_col_key), 1);
      $(this).removeClass('active');
      $(this).remove();

      $("#pred-input-cols div").each(function (index) {
        if(this.id=="form-"+input_col_key){
          $(this).remove();
        }
      });
    }
  });

  

  if (input_cols.length == 0) {
    $('#input-cols').html(`<div class="alert alert-info" role="alert">
            No input column has been selected.
          </div>`);
  }
}