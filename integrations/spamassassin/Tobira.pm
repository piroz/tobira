package Mail::SpamAssassin::Plugin::Tobira;

use strict;
use warnings;

use Mail::SpamAssassin::Plugin;
use Mail::SpamAssassin::Logger;

our @ISA = qw(Mail::SpamAssassin::Plugin);

my $PLUGIN_VERSION = "0.1.0";

sub new {
    my ($class, $mailsa) = @_;
    my $self = $class->SUPER::new($mailsa);
    bless($self, $class);

    $self->set_config($mailsa->{conf});
    $self->register_eval_rule("check_tobira");

    dbg("tobira: plugin v$PLUGIN_VERSION loaded");
    return $self;
}

sub set_config {
    my ($self, $conf) = @_;

    my @cmds = (
        {
            setting  => 'tobira_url',
            default  => 'http://127.0.0.1:8000/predict',
            type     => $Mail::SpamAssassin::Conf::CONF_TYPE_STRING,
        },
        {
            setting  => 'tobira_timeout',
            default  => 5,
            type     => $Mail::SpamAssassin::Conf::CONF_TYPE_NUMERIC,
        },
    );

    $conf->{parser}->register_commands(\@cmds);
}

sub check_tobira {
    my ($self, $pms) = @_;

    my $url     = $pms->{conf}->{tobira_url};
    my $timeout = $pms->{conf}->{tobira_timeout};

    my $body_ref = $pms->get_decoded_stripped_body_text_array();
    my $subject  = $pms->get("Subject:value") || "";
    my $text     = $subject . "\n" . join("", @{$body_ref || []});

    my $trimmed = $subject . join("", @{$body_ref || []});
    $trimmed =~ s/\s+//g;
    if (length($trimmed) == 0) {
        dbg("tobira: empty message body, skipping");
        return 0;
    }

    my ($label, $score) = $self->_call_api($url, $timeout, $text);
    unless (defined $score) {
        dbg("tobira: API call failed, inserting TOBIRA_FAIL");
        $pms->got_hit("TOBIRA_FAIL", "", score => 0);
        return 0;
    }

    if ($score >= 0.9) {
        $pms->got_hit("TOBIRA_SPAM_HIGH", "", ruletype => "header");
    } elsif ($score >= 0.7) {
        $pms->got_hit("TOBIRA_SPAM_MED", "", ruletype => "header");
    } elsif ($score >= 0.5) {
        $pms->got_hit("TOBIRA_SPAM_LOW", "", ruletype => "header");
    } elsif ($score < 0.3) {
        $pms->got_hit("TOBIRA_HAM", "", ruletype => "header");
    }

    $pms->set_tag("TOBIRALABEL", $label);
    $pms->set_tag("TOBIRASCORE", sprintf("%.4f", $score));

    return 0;
}

sub _call_api {
    my ($self, $url, $timeout, $text) = @_;

    eval { require LWP::UserAgent; require HTTP::Request; require JSON; };
    if ($@) {
        dbg("tobira: missing required module: $@");
        return (undef, undef);
    }

    my $ua = LWP::UserAgent->new(
        timeout => $timeout,
        agent   => "SpamAssassin-Tobira/$PLUGIN_VERSION",
    );

    my $payload = JSON::encode_json({ text => $text });

    my $req = HTTP::Request->new('POST', $url);
    $req->header('Content-Type' => 'application/json');
    $req->content($payload);

    my $res = eval { $ua->request($req) };
    if ($@ || !$res) {
        dbg("tobira: API request failed: " . ($@ || "no response"));
        return (undef, undef);
    }

    if (!$res->is_success) {
        dbg("tobira: API returned HTTP " . $res->code);
        return (undef, undef);
    }

    my $data = eval { JSON::decode_json($res->decoded_content) };
    if ($@ || !$data) {
        dbg("tobira: failed to parse JSON response: $@");
        return (undef, undef);
    }

    my $label = $data->{label};
    my $score = $data->{score};

    unless (defined $label && defined $score) {
        dbg("tobira: unexpected response format");
        return (undef, undef);
    }

    dbg("tobira: label=$label score=$score");
    return ($label, $score);
}

1;
